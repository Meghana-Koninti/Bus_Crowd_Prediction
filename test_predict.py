import sys
sys.path.insert(0, '.')
from app.services.prediction_service import predict_crowd

cases = [
    {'route': '9X',  'stop': 'Miyapur',  'hour': '10', 'day': 'Monday', 'weather': 'Clear',   'buses': 6, 'month': 5},
    {'route': '9X',  'stop': 'Miyapur',  'hour': '8',  'day': 'Monday', 'weather': 'Sunny',   'buses': 6, 'month': 5},
    {'route': '47L', 'stop': 'Ameerpet', 'hour': '17', 'day': 'Friday', 'weather': 'Rainy',   'buses': 5, 'month': 4},
    {'route': '218', 'stop': 'Uppal',    'hour': '3',  'day': 'Sunday', 'weather': 'Overcast','buses': 2, 'month': 7},
]

for c in cases:
    ok, result = predict_crowd(c)
    print(f"Route={c['route']}, Stop={c['stop']}, Hour={c['hour']}, Weather={c['weather']}")
    print(f"  -> OK={ok}, Result={result}")
