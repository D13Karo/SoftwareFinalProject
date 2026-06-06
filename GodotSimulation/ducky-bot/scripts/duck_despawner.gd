extends Node
# Press SPACEBAR to despawn the next duck in the list.
# Used to demonstrate the OBSTACLE_PRESENT -> LANE_FOLLOWING recovery
# transition for the object_detection task.

@export var duck_paths: Array[NodePath] = []
var _next_index: int = 0

func _ready() -> void:
	print("[Despawner] Ready. Press SPACE to remove the next duck. ",
		duck_paths.size(), " ducks queued.")

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		if event.keycode == KEY_SPACE:
			_despawn_next()

func _despawn_next() -> void:
	while _next_index < duck_paths.size():
		var path: NodePath = duck_paths[_next_index]
		_next_index += 1
		var node := get_node_or_null(path)
		if node:
			print("[Despawner] Removing ", node.name)
			node.queue_free()
			return
	print("[Despawner] No more ducks to remove.")
