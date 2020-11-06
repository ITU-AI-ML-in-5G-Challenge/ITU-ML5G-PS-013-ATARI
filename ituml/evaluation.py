import pandas as pd

import torch
import torch.nn.functional as F

########################################################################################################################
# Utilities.
########################################################################################################################


def access_point_throughputs(predictions, labels):
    predictions = predictions.squeeze().detach().to('cpu').numpy()
    labels = labels.squeeze().detach().to('cpu').numpy()
    df = pd.DataFrame({"predictions": predictions, "labels": labels})
    res = df.groupby("labels").sum().values
    res = torch.tensor(res, dtype=torch.float)
    return res


########################################################################################################################
# Evaluation.
########################################################################################################################


def scores(data, y_pred):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Extract station predictions and corresponding access point labels.
    station_predictions = y_pred[data.y_mask]
    station_labels = data.node_ap[data.y_mask]

    # Aggregate (sum) station predictions per access point and update model predictions.
    ap_predictions = access_point_throughputs(station_predictions, station_labels).to(device)
    y_pred[~data.y_mask] = ap_predictions

    # Calculate updated loss.
    rmse = torch.sqrt(F.mse_loss(y_pred.squeeze(), data.y))

    return rmse
