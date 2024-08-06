from audiosample import AudioSample

def register_plugin(func_name, func):
    """
    Register a plugin with AudioSample.
    """

    setattr(AudioSample, func_name, func)

def deregister_plugin(func_name):
    """
    Deregister a plugin with AudioSample.
    """

    delattr(AudioSample, func_name)

AudioSample.register_plugin = register_plugin
AudioSample.deregister_plugin = deregister_plugin
