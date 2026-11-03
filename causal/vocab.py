"""Canonical causal-node vocabulary shared by the Muse Spark teacher and the fast encoder.

Texts are generic (no object names) so the encoder can emit them without generating text; the
edit instruction supplies the object context in the final prompt.
"""

# id: (true_text, false_text)
NODES = {
    # power
    "mains_power": ("the wall outlet has power", "there is a power outage"),
    "plugged_in": ("the device is plugged into a power outlet", "the device's plug is not in any outlet"),
    "batteries_inside": ("the batteries are inside the device", "the batteries are not inside the device"),
    "batteries_charged": ("the batteries are charged", "the batteries are dead"),
    "power_switch_on": ("the power switch is on", "the power switch is off"),
    "device_powered": ("the device is receiving power", "the device has no power"),
    "device_on": ("the device is on and running", "the device stays off"),
    "bulb_intact": ("the bulb is intact", "the bulb is broken"),
    "light_emitted": ("it is emitting light and glowing", "it is dark and emits no light"),
    "screen_on": ("the screen is on, showing a picture", "the screen is black"),
    "motor_running": ("the motor is running and moving parts are in motion", "the moving parts are still"),
    "element_hot": ("the heating element is hot and glowing orange", "the heating element is cold"),
    "indicator_on": ("the indicator light is lit", "the indicator light is off"),
    "surroundings_lit": ("nearby surfaces are lit by its light", "the surroundings stay dim"),
    # liquid / water
    "water_supply_on": ("the water supply is on", "the water supply is shut off"),
    "hose_connected": ("the hose is connected to the tap", "the hose is disconnected from the tap"),
    "valve_open": ("the valve or tap is open", "the valve or tap is closed"),
    "container_has_liquid": ("the container holds liquid", "the container is empty"),
    "container_open": ("the container is open", "the container is sealed closed"),
    "container_tipped": ("the container is tipped over or tilted", "the container stands upright"),
    "container_intact": ("the container is intact", "the container is cracked or broken"),
    "liquid_flowing": ("liquid is flowing out", "no liquid flows out"),
    "liquid_spilled": ("liquid has spilled onto the surface", "the surface stays clean and dry"),
    "drain_open": ("the drain is open", "the drain is plugged"),
    "basin_overflowing": ("the basin overflows", "the basin does not overflow"),
    # heat / fire
    "fuel_available": ("fuel is available", "there is no fuel"),
    "wick_present": ("there is a wick", "there is no wick"),
    "ignition_applied": ("a flame or spark is applied", "no flame or spark is applied"),
    "dry_fuel": ("the fuel is dry", "the fuel is wet"),
    "flame_present": ("a flame is burning", "there is no flame"),
    "smoke_present": ("smoke rises", "there is no smoke"),
    "heat_applied": ("heat is applied", "no heat is applied"),
    "contents_heated": ("the contents are heated, steaming or boiling", "the contents stay cold"),
    "food_cooked": ("the food is cooked", "the food stays raw"),
    "ice_melted": ("the ice has melted", "the ice stays frozen"),
    # support / gravity / mechanics
    "object_supported": ("the object is supported", "the object is unsupported"),
    "object_secured": ("the object is fastened or tied down", "the object is loose"),
    "object_falls": ("the object falls", "the object stays in place"),
    "object_fragile": ("the object is fragile", "the object is sturdy"),
    "landing_soft": ("it lands on a soft surface", "it lands on a hard surface"),
    "object_breaks": ("the object breaks", "the object stays intact"),
    "chain_attached": ("the chain or belt is attached", "the chain or belt is off"),
    "force_applied": ("force is applied to it", "no force is applied"),
    "wheel_turning": ("the wheel turns", "the wheel stays still"),
    "lock_engaged": ("it is locked", "it is unlocked"),
    "door_open": ("the door or lid is open", "the door or lid stays closed"),
    "rope_intact": ("the rope or string is intact", "the rope or string is cut"),
    "wind_present": ("there is wind", "the air is still"),
    "object_moves": ("the object moves", "the object stays still"),
}
IDS = list(NODES)
FAMILIES = ["electrical", "liquid", "heat", "mechanical"]


def text(node, value):
    return NODES[node][0 if value else 1]

# Concrete visual descriptions for nodes whose state can be read from pixels, used for zero-shot state
# reading with the frozen image-text encoder. Both states are described as visible things, without
# negation ("not", "no"), which CLIP-style encoders handle poorly. Object-agnostic on purpose.
VISUAL = {
    "plugged_in": ("a power plug pushed into a wall socket", "a power plug lying loose on a surface"),
    "batteries_inside": ("a closed battery compartment with batteries inside", "batteries lying loose outside the device with the battery cap off"),
    "power_switch_on": ("a switch flipped to the ON position", "a switch flipped to the OFF position"),
    "bulb_intact": ("an intact glass light bulb", "a shattered broken light bulb"),
    "device_on": ("a switched-on device with its lights glowing", "a switched-off dark device"),
    "light_emitted": ("a glowing light source casting bright light", "a dark unlit light source"),
    "screen_on": ("a bright screen showing a colorful picture", "a black blank screen"),
    "motor_running": ("fast-spinning blades blurred with motion", "still motionless blades"),
    "element_hot": ("heating coils glowing bright orange", "dull grey cold heating coils"),
    "indicator_on": ("a small indicator light glowing", "a small dark indicator light"),
    "water_supply_on": ("a water valve with a green ON tag and its handle open", "a water valve with a red OFF tag and its handle closed"),
    "hose_connected": ("a garden hose screwed onto a tap", "a garden hose end lying on the ground away from the tap"),
    "valve_open": ("a tap with its handle turned open", "a tap with its handle turned fully closed"),
    "container_has_liquid": ("a transparent container full of liquid", "an empty transparent container"),
    "container_open": ("an open container with its cap or lid removed", "a container with its cap or lid tightly closed"),
    "container_tipped": ("a container lying tipped over on its side", "a container standing upright"),
    "container_intact": ("an intact container", "a cracked broken container"),
    "liquid_flowing": ("a stream of liquid pouring out", "a completely dry spout"),
    "liquid_spilled": ("a puddle of spilled liquid on the surface", "a clean dry surface"),
    "drain_open": ("an open drain hole", "a drain closed with a rubber stopper"),
    "basin_overflowing": ("water overflowing the basin rim", "water sitting below the basin rim"),
    "fuel_available": ("a fireplace or grill stacked with wood or fuel", "an empty bare fireplace or grill"),
    "wick_present": ("a candle with a visible wick", "a candle with a flat bare top"),
    "dry_fuel": ("dry wood", "soaking wet wood"),
    "flame_present": ("a burning flame", "a cold unlit burner"),
    "smoke_present": ("smoke rising", "clear air"),
    "heat_applied": ("a lit burner heating a pot", "a cold unlit burner under a pot"),
    "contents_heated": ("steam rising from hot contents", "cold still contents"),
    "food_cooked": ("cooked golden-brown food", "raw uncooked food"),
    "ice_melted": ("a puddle of melted water", "solid ice cubes"),
    "object_supported": ("an object resting firmly on a support", "an object hanging over an edge with empty space beneath it"),
    "object_secured": ("an object tied down with rope or straps", "a loose object resting freely"),
    "object_fragile": ("a fragile glass or ceramic object", "a sturdy rubber or metal object"),
    "landing_soft": ("a soft cushion or carpet below", "a hard concrete or tile floor below"),
    "object_breaks": ("shattered pieces scattered on the floor", "an intact undamaged object"),
    "chain_attached": ("a bicycle chain seated on the gears", "a bicycle chain hanging loose off the gears"),
    "wheel_turning": ("a wheel blurred with spinning motion", "a still wheel"),
    "lock_engaged": ("a padlock locked shut", "an open unlocked padlock"),
    "door_open": ("a door or lid standing open", "a door or lid shut closed"),
    "rope_intact": ("an intact taut rope", "a rope cut into two loose ends"),
    "wind_present": ("trees and flags bending in strong wind", "calm still air with limp flags"),
}
