# libgpteng

`libgpteng` is the engine library. The sibling [`gpteng`](../gpteng) project is
the native desktop client and process host. The engine library owns simulation
state, rendering and scripting APIs, plus the shared engine tool catalog; the
host owns desktop startup and MCP transport selection.

The initial implementation is organized around these boundaries:

- `engine`: fixed-step simulation, ECS-backed entities, parent-relative and
  world transforms, names, mesh/material components, and entity-attached lights.
- `render`: wgpu device setup and a forward 3D renderer with cached mesh
  instances, configurable scene lighting, conservative frustum culling, and
  per-frame submission statistics. A host supplies
  a window or browser canvas and configures the surface; the scene renderer
  draws engine meshes with perspective or orthographic projection, depth testing, and configurable
  directional and ambient lighting.
- `scripting`: bounded Lua runtimes plus a native worker that owns persistent
  entity scripts and advances their fixed-step callbacks.
- `control`: transport-neutral operations for inspecting and changing engine
  state. These APIs are intended to be shared by gameplay code, tools, and MCP.
- `mcp`: native MCP stdio and Streamable HTTP servers, plus the optional
  browser WebMCP page exposing entity inspection and mutation.

The desktop app renders a lit, animated Donald Trump caricature head built from
reusable engine primitives. The browser demo uses the same model and Lua head
motion. `spawn_donald_trump_caricature` returns the head root and face entity so
callers can add scripts and materials. Built-in meshes currently include cubes,
UV spheres, and horizontal planes. `spawn` creates a cube by default, and
`spawn_mesh`/`spawn_empty` allow engine callers to add geometry or transform-only
entities. Built-in and imported GPU meshes are
cached, and instances are batched by mesh asset. The bounded OBJ importer
supports positions, optional texture coordinates and normals, and polygon
faces; OBJ material libraries are ignored. The glTF 2.0/GLB importer handles
static triangle primitives, embedded buffers, editable node transforms and hierarchy,
material base color/roughness/metallic factors, embedded base-color images, and
node translation/rotation/scale animation with STEP, LINEAR, and CUBICSPLINE
interpolation. The first imported clip plays automatically; animation state is
saved in scene documents and can be controlled through engine APIs, Lua, MCP,
WebMCP, and the browser host. It rejects external resource references. glTF
OPAQUE, MASK, and BLEND alpha modes are imported; normal/metallic-roughness maps,
skins, and morph-target animation are not imported yet. Texture coordinates are carried
into the GPU vertex stream and sampled from repeating, linearly filtered sRGBA8
base-color textures. Texture assets are scene-persistent sRGBA8 data with
dimensions capped at 4096 pixels and a 16 MiB decoded-data budget per scene.
The engine also decodes bounded PNG, JPEG, WebP, BMP, GIF, TIFF, and PNM images;
animated formats use their first frame. Normal maps and image-based environment
lighting remain to be built. Materials carry linear RGBA base color, roughness,
and metallic response. Materials support automatic alpha handling plus
explicit opaque, masked, and blended modes. Masked fragments use an alpha
cutoff; blended materials use a depth-tested, back-to-front pass. Intersecting
translucent surfaces still use object-center sorting. The desktop and browser
demo include a translucent orb to exercise this path. Scenes support a
directional light and up to eight
entity-attached point lights per rendered frame; excess point lights are ranked
by an approximate camera-space contribution score. Entities can be parented
into a transform
hierarchy; rendering composes their full world matrices, parent deletion leaves
children alive as roots at their previous world poses, and cycle-forming
reparent operations are rejected. Visibility is saved with the scene and
inherited through the hierarchy, so hiding a root also hides its descendants.
Environment lighting, continuous collision detection, and editor tools remain
to be built. The simulation now integrates dynamic,
kinematic, and static rigid bodies at fixed steps, with gravity, accumulated
forces, velocity, and linear damping. Sphere and box colliders use a deterministic
sweep-and-prune broad phase followed by a discrete world-space overlap pass with
positional correction, restitution, and friction; rotated boxes are currently
approximated by their world-aligned bounds. Sparse scenes prune efficiently;
dense overlapping scenes can still require quadratic pair checks. The
active camera is stored in `Engine`, so desktop input and remote commands update
the same view.
The active camera is stored in `Engine`, so desktop input and remote commands update
the same view. Perspective remains the default; orthographic projection supports
isometric, strategy, and CAD-style framing. Camera projection and orthographic
vertical size are saved with the scene and available through MCP and WebMCP.
`Engine::to_scene_json` and `Engine::load_scene_json` save and atomically restore
versioned scene documents containing transforms, hierarchy, primitive and
imported meshes, materials, rigid bodies, colliders, gravity, camera, visibility,
and lighting state.
Native MCP and browser scene saves also contain attached Lua script sources; a
load runs each script's setup against the candidate scene before replacing the
live scene. Script VM globals and other runtime state start fresh.
Scene imports validate IDs, component values, asset references, and parent cycles
before replacing the active world. JSON input is capped at 64 MiB, with separate
entity, asset, mesh-element, and script-source limits. Lua VM state is host-owned
and is recreated from saved sources when a host loads the document.
Lua scripts run with per-call source, memory, and instruction
budgets. The native `ScriptWorker` keeps a Lua state per attached entity, runs
setup once, and calls an optional `update(dt)` on fixed simulation steps. The
desktop demo caricature uses this lifecycle to turn over time. `run_with_engine`
exposes `engine.list_entities()`, `engine.spawn(name, x, y, z, primitive)`,
`engine.set_position(id, x, y, z)`, `engine.delete(id)`, `engine.self()`, and
`engine.delta_time()`; scene mutations are committed through the same engine
API used by native gameplay and MCP. Lua also exposes
`engine.get_transform(id)`, `engine.set_rotation(id, x, y, z, w)`, and
`engine.set_scale(id, x, y, z)`, `engine.get_material(id)`, and
`engine.set_material(id, r, g, b, a, roughness, metallic)`. `engine.load_obj_mesh(name, source, x,
y, z)` imports bounded OBJ text and spawns a mesh; it is limited to 256 KiB.
`engine.load_gltf_mesh(name, source_base64, x, y, z)` imports embedded glTF/GLB
bytes from base64. Both importers allow four mesh loads per script run. Primitive
can be `cube`, `sphere`, or `plane`; a handle returned by `engine.spawn`,
`engine.load_obj_mesh`, or `engine.load_gltf_mesh` is valid within that
chunk, and later chunks can look up the assigned engine ID with
`engine.list_entities()`. Lua can inspect the active camera with
`engine.get_camera()` and change its position or target with
`engine.set_camera_position(x, y, z)` or `engine.set_camera_target(x, y, z)`.
`engine.set_camera_projection("perspective", 0)` or
`engine.set_camera_projection("orthographic", vertical_size)` changes projection;
the browser demo has a button that switches the animated caricature between both.
`engine.set_parent(child_id, parent_id)` reparents an entity, with `nil` as the
parent ID to make it a root, and `engine.get_world_transform(id)` returns its
composed world position, rotation, and scale.
Imported glTF clips can be controlled with
`engine.play_gltf_animation(root_id, clip_index, looping)`,
`engine.stop_gltf_animation(root_id)`, and
`engine.set_gltf_animation_speed(root_id, speed)`.
Lua lighting is available through `engine.get_lighting()` and
`engine.set_lighting(dx, dy, dz, r, g, b, intensity, ambient_r, ambient_g,
ambient_b)`.
Entity point lights use `engine.set_point_light(id, r, g, b, intensity,
radius)`, `engine.get_point_light(id)`, and `engine.remove_point_light(id)`.
Lua `engine.get_material(id)` reports roughness, metallic, and alpha settings;
`engine.set_material(id, r, g, b, a, roughness, metallic)` preserves alpha
settings while updating surface values. Use
`engine.set_alpha_mode(id, mode, cutoff)` with `auto`, `opaque`, `mask`, or
`blend` to change compositing behavior.
Lua scripts can also use `engine.add_rigid_body(id, body_type, mass,
gravity_scale, linear_damping)`, `engine.set_velocity(id, x, y, z)`,
`engine.apply_force(id, x, y, z)`, `engine.get_rigid_body(id)`,
`engine.remove_rigid_body(id)`, `engine.get_gravity()`, and
`engine.set_gravity(x, y, z)`.
Colliders are controlled with `engine.set_sphere_collider(id, radius,
restitution, friction)`, `engine.set_box_collider(id, half_x, half_y, half_z,
restitution, friction)`, `engine.get_collider(id)`, and
`engine.remove_collider(id)`.
Lua can inspect or assign an existing image with
`engine.get_base_color_texture(id)` and
`engine.set_base_color_texture(id, texture_asset_id_or_nil)`.
`engine.set_visibility(id, visible)`, `engine.get_visibility(id)`, and
`engine.is_visible(id)` update or inspect local and inherited render visibility.
Gameplay Lua can read host-supplied controls with `engine.is_key_down("w")`,
`engine.is_mouse_button_down("left")`, and `engine.mouse_delta()` (a two-value
array-like table). Hosts feed these through `Engine::set_key_down`,
`Engine::set_mouse_button_down`, and `Engine::add_mouse_motion`; input stays
transient and is omitted from saved scenes. Browser hosts can use the matching
`BrowserEngine` methods, and MCP/WebMCP expose the current state through
`get_input_state`.

The shared MCP and WebMCP tool catalog exposes `export_scene`, `load_scene`,
`list_entities`, `create_texture_asset`, `load_texture_asset`, `set_base_color_texture`,
`get_base_color_texture`, `set_visibility`, `get_visibility`, `get_lighting`,
`set_lighting`, `set_point_light`, `get_point_light`, `remove_point_light`,
`spawn_entity`,
`spawn_primitive`, `set_position`, `set_transform`, `set_material`, `get_material`,
`delete_entity`, `advance_simulation`, `get_engine_info`, `attach_script`,
`get_camera`, `set_camera`, `load_obj_mesh`, `load_gltf_mesh`, `detach_script`, `list_scripts`,
`set_parent`, `get_world_transform`, `add_rigid_body`, `get_rigid_body`,
`remove_rigid_body`, `set_velocity`, `apply_force`, `get_gravity`, `set_gravity`,
`set_collider`, `get_collider`, `remove_collider`, `list_gltf_animations`,
`play_gltf_animation`, `stop_gltf_animation`, `set_gltf_animation_speed`, and
`run_lua`. Both protocol
adapters use the same catalog and dispatcher. MCP `run_lua` uses a persistent, bounded Lua
state on the script worker; `attach_script` gives an entity its own persistent
state and optional `update(dt)` callback. The worker also advances simulation
steps in MCP-only/headless mode. `load_obj_mesh` accepts OBJ source text up to
8 MiB; `load_gltf_mesh` accepts base64-encoded glTF/GLB bytes up to 32 MiB after
decoding. It returns a root entity, node and primitive entity IDs, animation
names, and mesh asset IDs.

## Build targets

The engine library is designed to compile for `wasm32-unknown-unknown`. The
library also exports `BrowserEngine` on wasm32: it initializes a browser canvas
surface, advances the simulation and Lua scripts, and renders through wgpu.
It exposes `attach_script` and `detach_script` for persistent per-entity Lua,
along with keyboard, mouse-button, and pointer-motion input injection. The
browser example demonstrates a persistent script controlled by W/A/S/D and
creates a sampled checker texture from RGBA bytes.
`BrowserEngine.export_scene_json()` and `load_scene_json(json)` expose scene
persistence in the browser host as well. `BrowserEngine.load_gltf_mesh(name,
bytes, x, y, z)` imports embedded glTF or GLB bytes; the browser example builds
a minimal embedded glTF mesh to demonstrate the path. `set_alpha_mode(id, mode,
cutoff)` exposes alpha mode changes to browser clients. Browser clients can
inspect, start, stop, and change the speed of imported glTF clips through
`list_gltf_animations`, `play_gltf_animation`, `stop_gltf_animation`, and
`set_gltf_animation_speed`.
The example at `examples/wasm/index.html` exercises Rust scene creation and Lua.
Build it from this project root with
`wasm-pack build --target web --out-dir examples/wasm/pkg`, then serve this
directory over HTTP (for example, `python3 -m http.server 8000`) and open
`/examples/wasm/`. The renderer uses WebGPU when available and falls back to
wgpu's WebGL backend when the browser has no WebGPU adapter. The “Texture
caricature from image” control decodes a local PNG, JPEG, WebP, BMP, GIF, or
TIFF file into the model's face material.

The native `gpteng` host opens its desktop client by default and can also provide
stdio or Streamable HTTP MCP. Its `/webmcp` page registers the same shared Rust
tool catalog in a WebMCP-enabled browser. A bare
`wasm32-unknown-unknown` module has no operating-system socket or stdio service
to host an MCP server directly. In the desktop client, use W/A/S/D to move the
camera, E/Q to move vertically, hold the right mouse button to look around,
Shift to move faster, and Escape to close.

```sh
cargo check
cargo check --target wasm32-unknown-unknown
cargo run --manifest-path ../gpteng/Cargo.toml
cargo run --manifest-path ../gpteng/Cargo.toml -- --mcp-stdio
cargo run --manifest-path ../gpteng/Cargo.toml -- --mcp-http 127.0.0.1:8000 --webmcp
```

Lua currently uses Piccolo, a pure Rust Lua VM. This choice allows the same
runtime crate to build for native and `wasm32-unknown-unknown` and gives the
runtime instruction stepping needed for sandbox budgets. Piccolo is explicitly
experimental and does not implement all peripheral parts of PUC-Rio Lua's
standard library. The sandbox starts with only its core libraries enabled.
Enabling the `mcp` feature also enables `scripting-lua`, so every MCP build
publishes the Lua control tools alongside the direct engine tools.

## Workspace relationship

Keep engine mechanics, reusable rendering/scripting APIs, and engine control
tools in this project. Keep `gpteng` focused on native startup and host services.
The engine library should not depend on the sibling executable, window system,
or MCP transport.
