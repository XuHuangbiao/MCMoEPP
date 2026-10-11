class AverageMeter:
    def __init__(self, name=None, logger=None):
        self.name = name
        self.logger = logger
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        if hasattr(val, "detach"):
            val = float(val.detach().cpu())
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def log(self, step):
        if self.logger is not None:
            self.logger.add_scalar(self.name, self.avg, step)

    def done(self, step):
        self.log(step)
        ret = self.avg
        self.reset()
        return ret
