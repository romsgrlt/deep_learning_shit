import csv
import sys
import os

class Logger(object):
    def __init__(self, fpath=None, mode='w'):
        self.console = sys.stdout
        self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

class CSVBatchLogger:
    def __init__(self, csv_path, n_groups, mode='w'):
        columns = ['epoch', 'batch']
        for idx in range(n_groups):
            columns.append(f'avg_loss_group:{idx}')
            columns.append(f'exp_avg_loss_group:{idx}')
            columns.append(f'avg_acc_group:{idx}')
            columns.append(f'processed_data_count_group:{idx}')
            columns.append(f'update_data_count_group:{idx}')
            columns.append(f'update_batch_count_group:{idx}')
        columns.append('avg_actual_loss')
        columns.append('avg_per_sample_loss')
        columns.append('avg_acc')
        columns.append('model_norm_sq')
        columns.append('reg_loss')

        self.path = csv_path
        self.file = open(csv_path, mode)
        self.columns = columns
        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        if mode=='w':
            self.writer.writeheader()

    def log(self, epoch, batch, stats_dict):
        stats_dict['epoch'] = epoch
        stats_dict['batch'] = batch
        self.writer.writerow(stats_dict)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()