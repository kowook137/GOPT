
class Box(object):
    def __init__(
        self,
        length,
        width,
        height,
        x,
        y,
        z,
        orientation=0,
        box_type="unknown",
        box_type_id=0,
        box_id=-1,
    ):
        # dimension(x, y, z) + position(lx, ly, lz)
        self.size_x = length
        self.size_y = width
        self.size_z = height
        self.pos_x = x
        self.pos_y = y
        self.pos_z = z
        self.orientation = int(orientation)
        self.box_type = str(box_type)
        self.box_type_id = int(box_type_id)
        self.box_id = int(box_id)

    def standardize(self):
        """

        Returns:
            tuple(size + position)
        """
        return tuple([self.size_x, self.size_y, self.size_z, self.pos_x, self.pos_y, self.pos_z])
