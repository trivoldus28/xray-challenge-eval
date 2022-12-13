import itertools

import daisy
from daisy import Coordinate, Roi, Block


def get_chunks(
        block,
        chunk_div=None,
        chunk_shape=None):
    '''Convenient function to divide the given `block` into sub-blocks'''

    if chunk_shape is None:
        chunk_div = Coordinate(chunk_div)

        for j, k in zip(block.write_roi.get_shape(), chunk_div):
            assert (j % k) == 0

        chunk_shape = block.write_roi.get_shape() / Coordinate(chunk_div)

    else:
        chunk_shape = Coordinate(chunk_shape)

    write_offsets = itertools.product(*[
        range(
            block.write_roi.get_begin()[d],
            block.write_roi.get_end()[d],
            chunk_shape[d]
            )
        for d in range(chunk_shape.dims)
    ])

    context_begin = block.write_roi.get_begin()-block.read_roi.get_begin()
    context_end = block.read_roi.get_end()-block.write_roi.get_end()

    dummy_total_roi = Roi((0,)*chunk_shape.dims,
                          (0,)*chunk_shape.dims)
    chunk_roi = Roi((0,)*chunk_shape.dims, chunk_shape)
    blocks = []

    for write_offset in write_offsets:
        write_roi = chunk_roi.shift(Coordinate(write_offset)).intersect(
            block.write_roi)
        read_roi = write_roi.grow(context_begin, context_end).intersect(
            block.read_roi)

        if not write_roi.empty:
            chunk = Block(dummy_total_roi,
                          read_roi,
                          write_roi)
            blocks.append(chunk)

    return blocks


def enumerate_blocks_in_chunks(block, block_size, chunk_size, total_roi):

    if chunk_size is None:
        return block

    block_size = daisy.Coordinate(block_size)
    chunk_size = daisy.Coordinate(chunk_size)

    blocks = []
    chunk_shape = block_size / chunk_size
    ref_roi = daisy.Roi(block.write_roi.get_offset(), chunk_shape)

    offsets = [range(n) for n in chunk_size]

    for offset_mult in itertools.product(*offsets):

        shifted_roi = ref_roi.shift(chunk_shape*Coordinate(offset_mult))
        if total_roi.intersects(shifted_roi):
            blocks.append(
                daisy.Block(total_roi, shifted_roi, shifted_roi))

    return blocks
