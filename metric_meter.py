class MetricMeter:
    def __init__(self, beta=0.9):
        self.beta = beta
        self.ema_metric = None
    
    def update(self, current_metric):
        if self.ema_metric is None:
            self.ema_metric = current_metric
        else:
            self.ema_metric = self.beta * self.ema_metric + (1 - self.beta) * current_metric
    
    def get_ema_metric(self):
        return self.ema_metric

