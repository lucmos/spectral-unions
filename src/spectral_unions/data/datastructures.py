class Mesh:
    def __init__(
        self,
        *,
        v=None,
        f=None,
        mask=None,
        evals=None,
        raw_evals=None,
        name=None,
        area=None,
    ):
        self.v = v
        self.f = f
        self.area = area
        if f is not None:

            if f.min() == 1:
                self.f = self.f - 1
            elif f.min() == 0:
                pass
            else:
                raise RuntimeError(f"Invalid faces min value: {self.f}")

        self.eigs = self.evals = evals
        self.raw_evals = raw_evals
        # backward compatibility
        self.mask = self.indices = self.color = mask

        self.name = name
