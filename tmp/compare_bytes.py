if __name__ == '__main__':
    with open('gobytes.jpg', 'rb') as fgo:
        gobytes = fgo.read()

    with open('py_out.jpg', 'rb') as fpy: 
        pybytes = fpy.read()

    print(gobytes == pybytes)
    print(len(gobytes), len(pybytes))
