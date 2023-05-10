
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.acc_cur = 0.
        self.acc_avg = 0.
        self.correct_sum = 0
        self.count = 0

    def update(self, correct_cur, n=1):
        self.correct_sum += correct_cur
        self.count += n
        self.acc_cur = correct_cur / n
        self.acc_avg = self.correct_sum / self.count

