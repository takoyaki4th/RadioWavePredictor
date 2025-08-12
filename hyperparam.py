import time
from hyperparam_tuning.layer_unit_tuing import layer_unit_tuning

start_time=time.time()
layer_unit_tuning()
end_time=time.time()
print(f"実行時間:{(end_time-start_time):2f}秒")
