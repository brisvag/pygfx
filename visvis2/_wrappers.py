import ctypes

import wgpu
import numpy as np


def array_from_shader_type(spirv_type):
    """ Get a numpy array object from a SpirV type from python-shader.
    """
    return np.asarray(spirv_type())


# todo: can this be generic enough, keeping the GPU bits out / optional?

# todo: Support for updating unmapped data. Use something like updateRange to do subBufferUpdate


class BaseBufferWrapper:
    """ A base buffer wrapper that can be implemented for numpy, ctypes arrays,
    or any other kind of array.
    """

    def __init__(self, data=None, nbytes=None, usage=None, mapped=False):
        self._data = data
        if nbytes is not None:
            self._nbytes = int(nbytes)
        elif data is not None:
            self._nbytes = self._nbytes_from_data(data)
        else:
            raise ValueError("Buffer must be instantiated with either data or nbytes.")
        if isinstance(usage, int):
            self._usage = usage
        elif isinstance(usage, str):
            usages = usage.upper().replace(",", " ").replace("|", " ").split()
            assert usages
            self._usage = 0
            for usage in usages:
                self._usage |= getattr(wgpu.BufferUsage, usage)
        else:
            self._usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.VERTEX
            # raise ValueError("BufferWrapper usage must be int or str.")

        self._mapped = bool(mapped)
        self._dirty = True
        self._gpu_buffer = None  # Set by renderer

    @property
    def data(self):
        """ The data that is a view on the data. Can be None if the
        data only exists on the GPU.
        Note that this array can be replaced, so get it via this property.
        """
        # todo: maybe this class should not store _data if the data is not mapped?
        return self._data

    @property
    def nbytes(self):
        """ Get the number of bytes in the buffer.
        """
        return self._nbytes

    @property
    def usage(self):
        """ The buffer usage flags (as an int).
        """
        return self._usage

    @property
    def mapped(self):
        """ Whether the data is mapped. Mapped data can be updated in-place
        to change the data on the GPU.
        """
        return self._mapped

    @property
    def dirty(self):
        """ Whether the buffer is dirty (needs to be processed by the renderer).
        """
        return self._dirty

    @property
    def strides(self):
        """ Stride info (as a tuple).
        """
        return self._get_strides()

    @property
    def gpu_buffer(self):
        """ The WGPU buffer object. Can be None if the renderer has not set it (yet).
        """
        return self._gpu_buffer

    def set_mapped(self, mapped):
        self._mapped = bool(mapped)
        self._dirty = True

    def set_nbytes(self, n):
        self._nbytes = n
        self._dirty = True

    def set_data(self, data):
        """ Allow user to reset the array data.
        """
        self._data = data
        self._nbytes = self._nbytes_from_data(data)
        self._dirty = True

    def _renderer_set_gpu_buffer(self, buffer):
        # This is how the renderer marks the buffer as non-dirty
        self._gpu_buffer = buffer
        self._dirty = False

    # To implement in subclasses

    def _get_strides(self):
        raise NotImplementedError()

    def _nbytes_from_data(self, data):
        raise NotImplementedError()

    def _renderer_copy_data_to_ctypes_object(self, ob):
        """ Allows renderer to efficiently copy the data.
        """
        raise NotImplementedError()

    def _renderer_set_data_from_ctypes_object(self, ob):
        """ Allows renderer to replace the data.
        """
        raise NotImplementedError()

    def _renderer_get_data_dtype_str(self):
        """ Return numpy-ish dtype string, e.g. uint8, int16, float32.
        """
        raise NotImplementedError()

    def _renderer_get_vertex_format(self):
        raise NotImplementedError()


class BufferWrapper(BaseBufferWrapper):  # numpy-based
    """ Object that wraps a (GPU) buffer object, optionally providing data
    for it, and optionally *mapping* the data so it's shared. But you can also
    use it as a placeholder for a buffer with no representation on the CPU.
    """

    def _get_strides(self):
        return self.data.strides

    def _nbytes_from_data(self, data):
        return data.nbytes

    def _renderer_copy_data_to_ctypes_object(self, ob):
        ctypes.memmove(
            ctypes.addressof(ob), self.data.ctypes.data, self.nbytes,
        )

    def _renderer_set_data_from_ctypes_object(self, ob):
        new_array = np.asarray(ob)
        new_array.dtype = self._data.dtype
        new_array.shape = self._data.shape
        self._data = new_array

    def _renderer_get_data_dtype_str(self):
        return str(self.data.dtype)

    def _renderer_get_vertex_format(self):
        shape = self.data.shape
        if len(shape) == 1:
            shape = shape + (1,)
        assert len(shape) == 2
        key = str(self.data.dtype), shape[-1]
        mapping = {
            ("float32", 1): wgpu.VertexFormat.float,
            ("float32", 2): wgpu.VertexFormat.float2,
            ("float32", 3): wgpu.VertexFormat.float3,
            ("float32", 4): wgpu.VertexFormat.float4,
            #
            ("float16", 2): wgpu.VertexFormat.half2,
            ("float16", 4): wgpu.VertexFormat.half4,
            #
            ("int8", 2): wgpu.VertexFormat.char2,
            ("int8", 4): wgpu.VertexFormat.char4,
            ("uint8", 2): wgpu.VertexFormat.uchar2,
            ("uint8", 4): wgpu.VertexFormat.uchar4,
            #
            ("int16", 2): wgpu.VertexFormat.short2,
            ("int16", 4): wgpu.VertexFormat.short4,
            ("uint16", 2): wgpu.VertexFormat.ushort2,
            ("uint16", 4): wgpu.VertexFormat.ushort4,
            #
            ("int32", 1): wgpu.VertexFormat.int,
            ("int32", 2): wgpu.VertexFormat.int2,
            ("int32", 3): wgpu.VertexFormat.int3,
            ("int32", 4): wgpu.VertexFormat.int4,
            #
            ("uint32", 1): wgpu.VertexFormat.uint,
            ("uint32", 2): wgpu.VertexFormat.uint2,
            ("uint32", 3): wgpu.VertexFormat.uint3,
            ("uint32", 4): wgpu.VertexFormat.uint4,
        }
        try:
            return mapping[key]
        except KeyError:
            raise ValueError(f"Invalid dtype/shape for vertex data: {key}")
