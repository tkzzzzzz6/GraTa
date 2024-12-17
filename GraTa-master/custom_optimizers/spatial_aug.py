def augment(x, factor):
    # Flip
    if factor == 0:
        return x.flip(-1)  # horizontal
    elif factor == 1:
        return x.flip(-2)  # vertical
    # Rotate
    elif factor == 2:
        return x.flip(-1).transpose(-2, -1)  # 90
    elif factor == 3:
        return x.flip(-1).flip(-2)  # 180
    elif factor == 4:
        return x.transpose(-2, -1).flip(-1)  # 270
    # Identity
    elif factor == 5:
        return x
