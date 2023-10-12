from corax.raypad import address as rp_address


class RayAddressBuilder(rp_address.AbstractAddressBuilder):
    def __init__(self, name) -> None:
        super().__init__()
        self._name = name

    def build(self):
        return self._name
