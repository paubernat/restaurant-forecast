"""Adapters — concrete implementations of the ports. Only this layer does IO
and depends on third-party libraries (pandas readers, lightgbm, xgboost, timesfm,
mlflow). The domain never imports anything from here.
"""
