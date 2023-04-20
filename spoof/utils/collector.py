import json


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, name="", full=False):
        self.name = name
        self.reset()
        self.full = full

    def reset(self):
        self.vals = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.full:
            self.vals.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def mean(self):
        return self.avg


class MovingAverageMeter(AverageMeter):
    def __init__(self, name=None, full=False, momentum=0.9):
        super().__init__(name=name, full=full)
        self.momentum = momentum
        self.averaged_value = 0

    def update(self, vals: list):
        if self.full:
            self.vals.extend(vals)
        self.sum += sum(vals)
        self.count += len(vals)

        avg = sum(vals) / len(vals)
        if self.averaged_value == 0:
            self.averaged_value = avg
        else:
            self.averaged_value = (
                1 - self.momentum
            ) * avg + self.momentum * self.averaged_value

    @property
    def value(self):
        return self.averaged_value


class Collector:
    def __init__(self):
        self._dict = dict()

    def __getitem__(self, name):
        if name not in self._dict:
            self._dict[name] = AverageMeter(name)
        return self._dict[name]

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        s = "\n-> ".join(["Collector"] + [str(m) for m in self._dict.values()])
        return s

    def to_dict(self):
        return {name: meter.to_dict() for name, meter in self._dict.items()}

    def dump(self, file):
        with open(file, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class MovingAverageCollector(Collector):
    def __getitem__(self, name):
        if name not in self._dict:
            self._dict[name] = MovingAverageMeter(name, full=False)
        return self._dict[name]
