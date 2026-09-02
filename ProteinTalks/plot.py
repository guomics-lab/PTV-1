import os
import matplotlib.pyplot as plt

def plot_roc_curve(fpr, tpr, auroc, save_dir, title="ROC Curve"):
    """
    Plot ROC curve

    Args:
        fpr: False positive rates
        tpr: True positive rates
        auroc: AUROC value
        save_dir: Directory to save the plot
        title: Plot title
    """
    try:
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {auroc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300)
        plt.savefig(os.path.join(save_dir, 'roc_curve.pdf'))
        plt.close()

    except Exception as e:
        print(f"Error plotting ROC curve: {e}")

def plot_pr_curve(precision, recall, auprc, save_dir, title="Precision-Recall Curve"):
    """
    Plot Precision-Recall curve

    Args:
        precision: Precision values
        recall: Recall values
        auprc: AUPRC value
        save_dir: Directory to save the plot
        title: Plot title
    """
    try:
        plt.figure(figsize=(8, 8))
        plt.plot(recall, precision, 'g-', linewidth=2, label=f'PR curve (AP = {auprc:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)

        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'pr_curve.pdf'), dpi=300)
        plt.close()

    except Exception as e:
        print(f"Error plotting PR curve: {e}")
