import os
import pandas as pd
from datetime import datetime

class MetricsLogger:
    def __init__(self, output_dir):
        """Initialize metrics logger with output directory"""
        self.metrics_file = os.path.join(output_dir, 'metrics_log.csv')
        self.metrics_df = pd.DataFrame() if not os.path.exists(self.metrics_file) else pd.read_csv(self.metrics_file)
        
    def log_epoch_metrics(self, epoch, phase, metrics):
        """Log metrics for an epoch to CSV file"""
        # Flatten nested metrics dictionary
        flat_metrics = self._flatten_metrics_dict(metrics)
        
        # Add metadata
        flat_metrics.update({
            'epoch': epoch,
            'phase': phase,
            'timestamp': datetime.now().isoformat()
        })
        
        # Append to dataframe
        self.metrics_df = pd.concat([self.metrics_df, pd.DataFrame([flat_metrics])], ignore_index=True)
        self.metrics_df.to_csv(self.metrics_file, index=False)

    def get_performance_summary(self, last_n_epochs=None):
        """Get summary of key performance metrics"""
        df = self.metrics_df
        if last_n_epochs:
            df = df.sort_values('epoch').groupby('phase').tail(last_n_epochs)
            
        summary = {}
        for phase in ['train', 'val']:
            phase_df = df[df['phase'] == phase]
            if len(phase_df) > 0:
                summary[phase] = {
                    'avg_loss': phase_df['total_loss'].mean(),
                    'best_f1': phase_df['performance.f1_score'].max(),
                    'avg_precision': phase_df['performance.precision'].mean(),
                    'avg_recall': phase_df['performance.recall'].mean(),
                    'avg_iou': phase_df['iou_stats.mean'].mean()
                }
        
        return summary

    def plot_training_curves(self, metrics_to_plot=None):
        """Generate training curves for specified metrics"""
        if metrics_to_plot is None:
            metrics_to_plot = ['total_loss', 'performance.f1_score', 'iou_stats.mean']
            
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 4*len(metrics_to_plot)))
            if len(metrics_to_plot) == 1:
                axes = [axes]
                
            for ax, metric in zip(axes, metrics_to_plot):
                for phase in ['train', 'val']:
                    phase_data = self.metrics_df[self.metrics_df['phase'] == phase]
                    ax.plot(phase_data['epoch'], phase_data[metric], label=phase)
                ax.set_title(metric)
                ax.legend()
                ax.grid(True)
                
            plt.tight_layout()
            return fig
            
        except ImportError:
            print("matplotlib is required for plotting training curves")
            return None

    def _flatten_metrics_dict(self, metrics_dict, parent_key='', sep='.'):
        """Flatten nested dictionary into single level with dot notation keys, excluding distribution metrics"""
        items = []
        for k, v in metrics_dict.items():
            # Skip any keys containing 'distribution'
            if 'distribution' in k.lower():
                continue
                
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_metrics_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
                
        return dict(items)