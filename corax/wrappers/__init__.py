from corax.utils import lazy_loader

with lazy_loader.LazyImports(__name__, False):
    from corax.wrappers.gym_wrapper import GymWrapper
    from corax.wrappers.gymnasium_wrapper import GymnasiumWrapper

del lazy_loader
