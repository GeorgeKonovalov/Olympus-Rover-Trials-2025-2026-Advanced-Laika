import numpy as np
import trimesh
from manifold3d import Manifold, Mesh
from scipy.optimize import minimize_scalar

#Mostly AI generated

#Parameters

while True:
    try:
        kmin = float(input("Enter radius of drum\n"))         #Radial domain
        break
    except ValueError:
        print("Invalid input. Please enter a valid number for drum radius.")

while True:
    try:
        kmax = float(input("Enter radius of screw\n"))       #Radial domain
        break
    except ValueError:
        print("Invalid input. Please enter a valid number for screw radius.")

while True:
    try:
        Height = float(input("Enter height of screw\n"))     #Vertical domain
        break
    except ValueError:
        print("Invalid input. Please enter a valid number for screw height.")

while True:
    try:
        HelixAngle = float(input("Enter angle of screw (degrees)\n")) * np.pi / 180.0    
        break
    except ValueError:
        print("Invalid input. Please enter a valid number for screw angle.")

c = kmax * np.tan(HelixAngle) #Vertical rise per radian
print(f"Number of turns: {Height / (2 * np.pi * c):.2f}")
tmin, tmax = 0.0, Height / c  #Angle domain

while True:
    try:
        ThicknessMin = float(input("Enter minimum thickness of screw\n"))
        #Ensure thickness is less than pitch
        if ThicknessMin >= c * 2 * np.pi:
            print(f"Thickness ({ThicknessMin}) must be smaller than the helix pitch ({c * 2 * np.pi:.3f}).")
            continue
        break
    except ValueError:
        print("Invalid input. Please enter a valid number for thickness.")

while True:
    try:
        DrumTopExtension = float(input("Enter extension of drum above screw\n"))
        break
    except ValueError:
        print("Invalid input. Please enter a valid number for drum top extension.")

while True:
    try:
        DrumBottomExtension = float(input("Enter extension of drum below screw\n"))
        break
    except ValueError:
        print("Invalid input. Please enter a valid number for drum bottom extension.")

while True:
    try:
        SamplePerUnitT = int(input("Enter samples per unit angle (t)\n"))
        break
    except ValueError:
        print("Invalid input. Please enter a valid integer for samples per unit angle.")

while True:
    try:
        SamplePerUnitK = int(input("Enter samples per unit radius (k)\n"))
        break
    except ValueError:
        print("Invalid input. Please enter a valid integer for samples per unit radius.")

nt = int(SamplePerUnitT * (tmax - tmin))      #Samples along t
nk = int(SamplePerUnitK * (kmax - kmin))      #Samples along radius
wrap_t = False

#Gather both surface expressions together and validate in one block
while True:
    f_expr = input("Enter the offset of the lower surface as a function of k (defining the shape of the cross section)\nFor example 0 for a flat bottom or k**2 for a parabolic shape\n")
    g_expr = input("Enter the offset of the upper surface as a function of k (defin1ing the shape of the cross section)\n")
    try:
        #Basic syntax check
        test_k = 1.0
        eval(f_expr, {"np": np}, {"k": test_k})
        eval(g_expr, {"np": np}, {"k": test_k})
        #Additional geometry checks
        k_samples = np.linspace(kmin, kmax, 100)
        f_vals = [eval(f_expr, {"np": np}, {"k": kk}) for kk in k_samples]
        g_base_vals = [eval(g_expr, {"np": np}, {"k": kk}) for kk in k_samples]
        g_vals = [gb + ThicknessMin for gb in g_base_vals]
        pitch = c * 2 * np.pi
        if any(g < f + ThicknessMin for g, f in zip(g_vals, f_vals)):
            raise ValueError("Upper surface (including ThicknessMin) must be at least ThicknessMin above lower surface at all radii.")
        max_thickness = max(g - f for g, f in zip(g_vals, f_vals))
        if max_thickness >= pitch:
            raise ValueError(f"Thickness too large; may cause self-intersection (max thickness {max_thickness:.3f} >= pitch {pitch:.3f}).")
        #Passed all checks
        break
    except Exception as e:
        print(f"Invalid expression or geometry check failed: {e}. Please reenter both expressions.")

_eval_ns = {"np": np}

def f(k):
    return eval(f_expr, _eval_ns, {"k": k})

def g(k):
    base = eval(g_expr, _eval_ns, {"k": k})
    return base + ThicknessMin  #g is always above f by at least ThicknessMin

#Build sample grid

t = np.linspace(tmin, tmax, nt, endpoint=not wrap_t)
k = np.linspace(kmin, kmax, nk)
T, K = np.meshgrid(t, k, indexing='ij')

X1 = K * np.cos(T)
Y1 = K * np.sin(T)
Z1 = c * T + f(K)

X2 = K * np.cos(T)
Y2 = K * np.sin(T)
Z2 = c * T + g(K)

V1 = np.column_stack([X1.ravel(), Y1.ravel(), Z1.ravel()])
V2 = np.column_stack([X2.ravel(), Y2.ravel(), Z2.ravel()])

vertices = np.vstack([V1, V2])

#Helper to index (i,j) into flattened index for V1
def idx1(i, j):
    return i * nk + j

def idx2(i, j):
    return nt * nk + i * nk + j

faces = []

#Build faces on surface H1 and H2 (quads -> 2 triangles)
t_range = range(nt - 1) if not wrap_t else range(nt)
for i in t_range:
    inext = (i + 1) % nt
    for j in range(nk - 1):
        a = idx1(i, j)
        b = idx1(i, j + 1)
        c0 = idx1(inext, j)
        d = idx1(inext, j + 1)
        faces.append([a, b, c0])
        faces.append([b, d, c0])

        #Corresponding on top surface (V2), same connectivity but offset
        A = idx2(i, j)
        B = idx2(i, j + 1)
        C = idx2(inext, j)
        D = idx2(inext, j + 1)
        faces.append([A, C, B])   #Reverse winding so normals point outward consistently
        faces.append([B, C, D])

#Connect the two surfaces along k-edges (for every t sample, wrap along k)
#For each i and each j (including j = nk-1 -> we already connected interior cells,
#But we need the side walls at j=0 and j=nk-1 to close the volume)
#Choose a range of t‑indices depending on whether we are wrapping in t

if wrap_t:
    wall_range = range(nt)
else:
    wall_range = range(nt - 1)

for i in wall_range:
    inext = (i + 1) % nt if wrap_t else i + 1
    #Inner radial wall (j = 0) between V1 and V2
    j = 0
    a = idx1(i, j)
    b = idx1(inext, j)
    A = idx2(i, j)
    B = idx2(inext, j)
    faces.append([a, b, B])
    faces.append([a, B, A])

    #Outer radial wall (j = nk - 1)
    j = nk - 1
    a = idx1(i, j)
    b = idx1(inext, j)
    A = idx2(i, j)
    B = idx2(inext, j)
    faces.append([b, a, B])
    faces.append([a, A, B])

#Optionally, if t is not wrapped, you must cap the ends at i=0 and i=nt-1
if not wrap_t:
    #Cap at i=0
    i = 0
    #Create cap triangles between H1 row i and H2 row i
    for j in range(nk - 1):
        faces.append([ idx2(i, j), idx2(i, j+1), idx1(i, j) ])
        faces.append([ idx1(i, j), idx2(i, j+1), idx1(i, j+1) ])
    #Cap at i=nt-1
    i = nt - 1
    for j in range(nk - 1):
        faces.append([ idx1(i, j), idx1(i, j+1), idx2(i, j) ])
        faces.append([ idx2(i, j), idx1(i, j+1), idx2(i, j+1) ])

#Finalize mesh arrays
vertices = np.ascontiguousarray(vertices, dtype=np.float64)
faces = np.ascontiguousarray(np.array(faces, dtype=np.int64), dtype=np.int64)

mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

#Quick sanity: repair winding / normals (optional)
mesh.fix_normals()

#Convert to manifold3d

verts32 = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
faces32 = np.ascontiguousarray(mesh.faces, dtype=np.uint32)

verts32 = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
faces32 = np.ascontiguousarray(mesh.faces, dtype=np.uint32)

mesh_m = Mesh(verts32, faces32)
m = Manifold(mesh_m)

# Create central drum and union it robustly with the helicoid
#Add a tiny overlap to avoid degenerate touching that can break boolean ops

eps = max(1e-6, (kmax - kmin) * 1e-6)
drum_radius = kmin + eps

#Compute vertical span of the inner edge of the two surfaces
#First, find min of f(k) and max of g(k) over [kmin, kmax]
k_samples = np.linspace(kmin, kmax, 100)
f_vals = [f(kk) for kk in k_samples]
g_vals = [g(kk) for kk in k_samples]
min_f = min(f_vals)
max_g = max(g_vals)

z_base = c * tmin + min_f - DrumBottomExtension  #Lower z at the lowest point of lower surface
z_top = c * tmax + max_g + DrumTopExtension      #Upper z at the highest point of upper surface

drum_height = float(z_top - z_base)

drum = trimesh.creation.cylinder(radius=float(drum_radius), height=drum_height, sections=128)
#Position drum so its bottom matches z_base
drum.apply_translation([0.0, 0.0, z_base + drum_height / 2.0])
#Convert drum to manifold and union with main surface
drum_verts32 = np.ascontiguousarray(drum.vertices, dtype=np.float32)
drum_faces32 = np.ascontiguousarray(drum.faces, dtype=np.uint32)
drum_m = Manifold(Mesh(drum_verts32, drum_faces32))
m = m + drum_m  #Union the drum with the helicoid solid

#Convert back to trimesh for checks and export
mesh_final = trimesh.Trimesh(vertices=m.to_mesh().vert_properties,
                             faces=m.to_mesh().tri_verts,
                             process=False)

print("Watertight:", mesh_final.is_watertight)
print("Volume (mm^3):", mesh_final.volume)

#Export
mesh_final.export("ArchimedianScrew.stl")
print("Exported ArchimedianScrew.stl")
mesh_final.show()