# -*- coding: utf-8 -*-
"""UE UV Repair for Maya 2025. Commands perform all scene edits on a duplicate."""
import os
import traceback
from collections import defaultdict
from dataclasses import dataclass
from itertools import groupby

try:
    from maya import cmds, mel
    from maya.api import OpenMaya as om
except ModuleNotFoundError:
    om = None  # The edge matching algorithm also runs outside Maya.

VERSION = "1.0.2"
WINDOW = "UEUVRepair102Window"
MENU = "UEUVRepair102Menu"
UI = {}


def maya_useNewAPI():
    pass


@dataclass
class Face:
    verts: tuple
    ids: tuple
    sign: int
    material: int


def winding(coords):
    if not coords:
        return 0
    # Translate before calculating signed area for UVs far from the origin.
    x0, y0 = coords[0]
    area = sum((x-x0)*(v-y0) - (y-y0)*(u-x0)
               for (x, y), (u, v) in zip(coords, coords[1:] + coords[:1]))
    return (area > 0) - (area < 0)


class Mesh:
    def __init__(self, shape):
        sel = om.MSelectionList()
        sel.add(shape)
        self.path = sel.getDagPath(0)
        fn = om.MFnMesh(self.path)
        self.name = fn.currentUVSetName()
        if not self.name:
            raise ValueError("没有当前 UV 集")
        self.shells = fn.getUvShellsIds(self.name)[0]
        u, v = fn.getUVs(self.name)
        self.uv = list(zip(u, v))
        if not self.uv:
            raise ValueError("当前 UV 集为空")
        self.points = [(p.x, p.y, p.z) for p in fn.getPoints(om.MSpace.kObject)]
        sizes, verts = fn.getVertices()
        uv_sizes, uv_ids = fn.getAssignedUVs(self.name)
        _, materials = fn.getConnectedShaders(self.path.instanceNumber())
        self.faces = []
        start = uv_start = 0
        for i, size in enumerate(sizes):
            vid = tuple(verts[j] for j in range(start, start+size))
            count = uv_sizes[i]
            ids = tuple(uv_ids[j] for j in range(uv_start, uv_start+count))
            sign = winding([self.uv[j] for j in ids]) if count == size else 0
            self.faces.append(Face(vid, ids, sign, materials[i] if i < len(materials) else -1))
            start += size
            uv_start += count

    def edges(self):
        it = om.MItMeshEdge(self.path)
        while not it.isDone():
            yield it.index(), it.vertexId(0), it.vertexId(1), tuple(it.getConnectedFaces())
            it.next()


def side(face, a, b):
    """Return endpoint UV ids and the direction in the polygon winding."""
    if not face.sign or len(face.ids) != len(face.verts):
        return None
    i, j = face.verts.index(a), face.verts.index(b)
    if (i+1) % len(face.verts) == j:
        return face.ids[i], face.ids[j], 1
    if (j+1) % len(face.verts) == i:
        return face.ids[i], face.ids[j], -1
    return None


def uv_within_tolerance(u1, v1, u2, v2, tolerance):
    if tolerance == 0:
        return u1 == u2 and v1 == v2
    du, dv = u1 - u2, v1 - v2
    return du * du + dv * dv <= tolerance * tolerance


def compatible(mesh, first, second, tol, protect_material):
    f1, a, b = first
    f2, c, d = second
    one, two = mesh.faces[f1], mesh.faces[f2]
    if f1 == f2 or one.sign != two.sign or (protect_material and one.material != two.material):
        return False
    s, t = side(one, a, b), side(two, c, d)
    if not s or not t or s[2] == t[2]:
        return False
    for i, j in zip(s[:2], t[:2]):
        u1, v1 = mesh.uv[i]
        u2, v2 = mesh.uv[j]
        if not uv_within_tolerance(u1, v1, u2, v2, tol):
            return False
    return True


def make_plan(mesh, tol, protect_material, boundaries):
    seams, border = [], defaultdict(list)
    for eid, a, b, adj in mesh.edges():
        if len(adj) == 2:
            f, g = adj
            if compatible(mesh, (f, a, b), (g, a, b), tol, protect_material):
                if side(mesh.faces[f], a, b)[:2] != side(mesh.faces[g], a, b)[:2]:
                    seams.append(eid)
        elif boundaries and len(adj) == 1:
            p, q = mesh.points[a], mesh.points[b]
            if p == q:
                continue
            if p > q:
                p, q, a, b = q, p, b, a
            border[p, q].append((adj[0], a, b))

    # At a point, weld only a single connected fan of matched boundary edges.
    links = defaultdict(list)
    for pair in border.values():
        if len(pair) != 2 or not compatible(mesh, *pair, tol, protect_material):
            continue
        (_, a, b), (_, c, d) = pair
        for i, j in ((a, c), (b, d)):
            if i != j:
                links[mesh.points[i]].append((i, j))
    weld = {}
    for pairs in links.values():
        graph = defaultdict(set)
        for i, j in pairs:
            graph[i].add(j)
            graph[j].add(i)
        group, stack = set(), [next(iter(graph))]
        while stack:
            i = stack.pop()
            if i not in group:
                group.add(i)
                stack.extend(graph[i] - group)
        if len(group) == len(graph):
            rep = min(group)
            weld.update((i, rep) for i in group)
    return seams, weld


def expected_uv_groups(mesh, seams):
    """Predict all intended UV id unions; shared endpoints can join transitively."""
    roots = {}
    def representative(i):
        current = i
        while current in roots:
            current = roots[current]
        while i in roots:
            nxt = roots[i]
            roots[i] = current
            i = nxt
        return current
    wanted = set(seams)
    for eid, a, b, adjacent in mesh.edges():
        if eid not in wanted:
            continue
        f, g = adjacent
        for i, j in zip(side(mesh.faces[f], a, b)[:2], side(mesh.faces[g], a, b)[:2]):
            left, right = representative(i), representative(j)
            if left != right:
                roots[right] = left
    return {i: representative(i) for i in roots}


def check_uv_groups(before, after, groups):
    mapping, inverse = {}, {}
    for old, new in zip(before.faces, after.faces):
        if len(old.ids) != len(new.ids):
            raise RuntimeError("UV 面角数量改变")
        for i, j in zip(old.ids, new.ids):
            rep = groups.get(i, i)
            if mapping.setdefault(rep, j) != j or inverse.setdefault(j, rep) != rep:
                raise RuntimeError("缝合结果与选定边不一致")


def components(shape, kind, ids):
    parts = []
    for _, grouped in groupby(enumerate(sorted(set(ids))), lambda pair: pair[1]-pair[0]):
        values = [v for _, v in grouped]
        index = str(values[0]) if len(values) == 1 else "%d:%d" % (values[0], values[-1])
        parts.append("%s.%s[%s]" % (shape, kind, index))
    return parts


def primary_shape(root):
    shapes = cmds.listRelatives(root, shapes=True, noIntermediate=True, fullPath=True) or []
    if len(shapes) != 1 or cmds.nodeType(shapes[0]) != "mesh":
        raise ValueError("请选择只包含一个静态多边形形状的物体")
    return shapes[0]


def verify_topology(before, after, weld):
    if len(before.faces) != len(after.faces):
        raise RuntimeError("面数发生变化")
    mapping, inverse = {}, {}
    for old, new in zip(before.faces, after.faces):
        if len(old.verts) != len(new.verts):
            raise RuntimeError("面的顶点数发生变化")
        for v, w in zip(old.verts, new.verts):
            rep = weld.get(v, v)
            if mapping.setdefault(rep, w) != w or inverse.setdefault(w, rep) != rep:
                raise RuntimeError("顶点连接变化超出预计范围")
            if before.points[v] != after.points[w]:
                raise RuntimeError("模型顶点坐标发生变化")


def verify_uv_sets(source, target, active, tol):
    def fn(shape):
        sel = om.MSelectionList()
        sel.add(shape)
        return om.MFnMesh(sel.getDagPath(0))
    old, new = fn(source), fn(target)
    if set(old.getUVSetNames()) != set(new.getUVSetNames()):
        raise RuntimeError("UV 集名称发生变化")
    for name in old.getUVSetNames():
        oc, oi = old.getAssignedUVs(name)
        nc, ni = new.getAssignedUVs(name)
        if list(oc) != list(nc):
            raise RuntimeError("UV 分配发生变化")
        ou, ov = old.getUVs(name)
        nu, nv = new.getUVs(name)
        limit = tol if name == active else 0
        other_forward, other_back = {}, {}
        for a, b in zip(oi, ni):
            if not uv_within_tolerance(ou[a], ov[a], nu[b], nv[b], limit):
                raise RuntimeError("UV 坐标变化超过设定容差（%s）" % name)
            if name != active:
                if other_forward.setdefault(a, b) != b or other_back.setdefault(b, a) != a:
                    raise RuntimeError("其他 UV 集的缝发生变化（%s）" % name)


def run_item(root, repair, tol, protect, weld_enabled):
    shape = primary_shape(root)
    before = Mesh(shape)
    seams, weld = make_plan(before, tol, protect, True)
    desc = "%s  |  %s  |  %d 个壳，可缝 %d 条边，可焊 %d 个顶点" % (
        root.rsplit("|", 1)[-1], before.name, before.shells, len(seams), len(weld))
    if not repair or not (seams or (weld and weld_enabled)):
        return desc, None
    if cmds.listRelatives(root, children=True, type="transform"):
        raise ValueError("该网格下还有子物体，请单独选择叶级静态网格")
    parent = root.rsplit("|", 1)[0]
    short = root.rsplit("|", 1)[-1].split(":")[-1]
    copy = cmds.duplicate(root, name=short+"_UVfix#", returnRootsOnly=True)[0]
    copy = parent + "|" + copy.rsplit("|", 1)[-1]
    try:
        dest = primary_shape(copy)
        cmds.delete(copy, constructionHistory=True)
        dest = primary_shape(copy)
        cmds.polyUVSet(dest, currentUVSet=True, uvSet=before.name)
        mid = before
        if weld_enabled and weld:
            cmds.polyMergeVertex(components(dest, "vtx", weld), distance=1e-10,
                                 worldSpace=False, alwaysMergeTwoVertices=False,
                                 texture=False, constructionHistory=True)
            mid = Mesh(dest)
            verify_topology(before, mid, weld)
            seams, _ = make_plan(mid, tol, protect, False)
        if not seams:
            raise RuntimeError("没有可缝合的 UV 边")
        expected = expected_uv_groups(mid, seams)
        cmds.polyMapSew(components(dest, "e", seams), constructionHistory=True)
        after = Mesh(dest)
        verify_topology(before, after, weld if weld_enabled else {})
        verify_uv_sets(shape, dest, before.name, tol)
        check_uv_groups(mid, after, expected)
        return "%s\n  → %s，UV 壳 %d → %d，缝合 %d 条边" % (
            desc, copy, before.shells, after.shells, len(seams)), copy
    except Exception:
        if cmds.objExists(copy):
            cmds.delete(copy)
        raise


def run(repair):
    original = cmds.ls(selection=True, long=True) or []
    tol = cmds.floatField(UI["tol"], query=True, value=True)
    weld = cmds.checkBox(UI["weld"], query=True, value=True)
    protect = cmds.checkBox(UI["mat"], query=True, value=True)
    logs = ["UE UV 修复 %s | %s" % (VERSION, os.path.basename(__file__))]
    copies = []
    if not original or any("." in obj for obj in original):
        cmds.warning("请切换到物体模式并选择要处理的静态网格")
        return
    shapes = cmds.ls(original, dag=True, type="mesh", noIntermediate=True, long=True) or []
    roots = list(dict.fromkeys(s.rsplit("|", 1)[0] for s in shapes))
    if repair:
        cmds.undoInfo(openChunk=True, chunkName="UE UV Repair")
    cmds.waitCursor(state=True)
    try:
        for root in roots:
            try:
                msg, result = run_item(root, repair, tol, protect, weld)
                logs.append(msg)
                if result:
                    copies.append(result)
            except Exception as error:
                frame = traceback.extract_tb(error.__traceback__)[-1]
                logs.append("%s：跳过。%s\n出错位置：%s 第 %d 行，%s()" % (
                    root, error, os.path.basename(frame.filename), frame.lineno, frame.name))
        if repair:
            cmds.select(copies or original, replace=True)
        logs.append("已选择修复副本，可在视口按 Ctrl+1 隔离查看。" if copies
                    else ("未生成副本。" if repair else "检测完成。"))
        if not roots:
            logs.insert(0, "选择中没有多边形网格。")
        cmds.scrollField(UI["log"], edit=True, text="\n".join(logs))
    finally:
        cmds.waitCursor(state=False)
        if repair:
            cmds.undoInfo(closeChunk=True)


def show(*_):
    for window in ("UEUVRepairWindow", WINDOW):
        if cmds.window(window, exists=True):
            cmds.deleteUI(window, window=True)
    cmds.window(WINDOW, title="UE 碎 UV 修复  |  "+VERSION, widthHeight=(530, 365))
    cmds.columnLayout(adjustableColumn=True, rowSpacing=8, columnAttach=("both", 12))
    cmds.text(label="选中静态模型，在当前 UV 集检测或生成修复副本。", align="left")
    cmds.text(label="UV 坐标容差（0 表示完全重合）", align="left")
    UI["tol"] = cmds.floatField(value=0.0, minValue=0, maxValue=0.001, precision=7)
    UI["mat"] = cmds.checkBox(label="保留材质分界", value=True)
    UI["weld"] = cmds.checkBox(label="同时修复断开的几何顶点（仅限位置完全重合的对应边）", value=False)
    cmds.button(label="检测选中模型", height=32, command=lambda *_: run(False))
    cmds.button(label="生成修复副本", height=38, command=lambda *_: run(True))
    UI["log"] = cmds.scrollField(editable=False, wordWrap=True, height=140,
                                 text="当前版本 "+VERSION+"。先选原模型，再点检测。")
    cmds.showWindow(WINDOW)


def onMayaDroppedPythonFile(*_):
    # Call this file's implementation directly, even if an older plug-in is loaded.
    show()


if om is not None:
    class OpenWindow(om.MPxCommand):
        def doIt(self, args):
            show()


def initializePlugin(obj):
    plugin = om.MFnPlugin(obj, "UEUVRepair", VERSION, "Any")
    plugin.registerCommand("ueUVRepair102", OpenWindow)
    if not cmds.about(batch=True):
        main = mel.eval('$tmp = $gMainWindow')
        if cmds.menu(MENU, exists=True):
            cmds.deleteUI(MENU)
        cmds.menu(MENU, parent=main, label="UE UV 1.0.2")
        cmds.menuItem(parent=MENU, label="碎 UV 修复…", command=show)


def uninitializePlugin(obj):
    if not cmds.about(batch=True):
        if cmds.window(WINDOW, exists=True):
            cmds.deleteUI(WINDOW)
        if cmds.menu(MENU, exists=True):
            cmds.deleteUI(MENU)
    om.MFnPlugin(obj).deregisterCommand("ueUVRepair102")


if __name__ == "__main__" and om is not None:
    show()
