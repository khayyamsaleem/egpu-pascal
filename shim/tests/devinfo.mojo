from max.gpu.host import DeviceContext

def main() raises:
    print("has accelerator:", DeviceContext.number_of_devices())
    var ctx = DeviceContext()
    print("device:", ctx.name())
