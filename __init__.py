"""ComfyUI-CausalForcing: unofficial ComfyUI integration of Causal Forcing and Causal Forcing++ (thu-ml)."""

# ComfyUI imports this file as a package. pytest also imports it, as a bare
# module without a parent package, while collecting the repository root;
# the relative import is only meaningful in the first case.
if __package__:
    from .causalforcing_comfy.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
