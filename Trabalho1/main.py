import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import sys
from scipy.fftpack import idct, dct
import cv2
import os
from PIL import Image

ToYCbCr = np.matrix([[0.299, 0.587, 0.114],
                     [-0.168736, -0.331264, 0.5],
                     [0.5, -0.418688, -0.081312]])

ToRGB = np.linalg.inv(ToYCbCr)

Q_Y = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99]])

Q_CbCr = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                    [18, 21, 26, 66, 99, 99, 99, 99],
                    [24, 26, 56, 99, 99, 99, 99, 99],
                    [47, 66, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99]])

qf = 75
def colorMap(color_cmap):
    if color_cmap.lower() == "red":
        cmap = colors.LinearSegmentedColormap.from_list(color_cmap, [(0, 0, 0), (1, 0, 0)], 256)
    elif color_cmap.lower() == "gray":
        cmap = colors.LinearSegmentedColormap.from_list(color_cmap, [(0, 0, 0), (1, 1, 1)], 256)
    elif color_cmap.lower() == "blue":
        cmap = colors.LinearSegmentedColormap.from_list(color_cmap, [(0, 0, 0), (0, 0, 1)], 256)
    elif color_cmap.lower() == "green":
        cmap = colors.LinearSegmentedColormap.from_list(color_cmap, [(0, 0, 0), (0, 1, 0)], 256)
    else:
        sys.stderr.write("Error: Undefined colormap.\n")
        return

    return cmap


def show_image(title, title_color, color, img):
    plt.figure()
    #plt.close("all")
    #plt.axis("off")
    if color.lower() == "none":
        plt.imshow(img)
    else:
        plt.imshow(img, colorMap(color))

    plt.title(title, color=title_color)
    plt.show()


# ------------------- Exercicio 1 ----------------------------

def compress_3ways(path, name):
    save_file = path + name
    img = Image.open(path + name + ".bmp")
    img = img.convert("RGB")
    img.save(save_file + "-high.jpg", quality=75)
    img.save(save_file + "-medium.jpg", quality=50)
    img.save(save_file + "-low.jpg", quality=25)
    size_o = os.stat(save_file + ".bmp").st_size
    size_h = os.stat(save_file + "-high.jpg").st_size
    size_m = os.stat(save_file + "-medium.jpg").st_size
    size_l = os.stat(save_file + "-low.jpg").st_size
    print(name + " -> " + str(size_o))
    print("High -> " + str(size_h) + " || Taxa de Compressão -> " + str(round((1 - (size_h/size_o)) * 100, 2)) + "%")
    print("Medium -> " + str(size_m) + " || Taxa de Compressão -> " + str(round((1 - (size_m/size_o)) * 100, 2)) + "%")
    print("Low -> " + str(size_l) + " || Taxa de Compressão -> " + str(round((1 - (size_l/size_o)) * 100, 2)) + "%")
    print("\n")


# ------------------- Exercicio 3 ----------------------------

def split_channel(img):
    red_channel = img[:, :, 0]
    show_image("img_red", "gray", "red", red_channel)

    green_channel = img[:, :, 1]
    show_image("img_green", "gray", "green", green_channel)

    blue_channel = img[:, :, 2]
    show_image("img_blue", "gray", "blue", blue_channel)

    return [red_channel, green_channel, blue_channel, img.shape]


def reverse_split(array):
    nl, nc = array[0].shape
    img_rec = np.zeros((nl, nc, 3)).astype(np.uint8)
    img_rec[:, :, 0] = array[0]
    img_rec[:, :, 1] = array[1]
    img_rec[:, :, 2] = array[2]

    #show_image("img_rec", "gray", "None", img_rec)

    return img_rec


# ------------------- Exercicio 4 ----------------------------

def padding(img):
    shape = img.shape
    lines = shape[0]
    columns = shape[1]

    if shape[0] % 32 != 0:
        lines = 32 - shape[0] % 32
    if shape[1] % 32 != 0:
        columns = 32 - shape[1] % 32
    new = img

    lastL = new[-1, :][np.newaxis, :]
    addLines = lastL.repeat(lines, axis=0)
    new = np.vstack((new, addLines))

    lastC = new[:, -1][:, np.newaxis]
    addColumns = lastC.repeat(columns, axis=1)

    new = np.hstack((new, addColumns))

    show_image("Padding", "gray", "None", new)

    return new


def reverse_padding(img_rec, img):
    nl, nc, nch = img.shape
    img_final = img_rec[:nl, :nc]
    show_image("RevPadding", "gray", "None", img_final)

    return img_final


# ------------------- Exercicio 5 ----------------------------

def to_YCbCr(r, g, b, shape):
    # Converter de RGB para yCbCr

    Y = (ToYCbCr[0, 0] * r + ToYCbCr[0, 1] * g + ToYCbCr[0, 2] * b)
    Cb = (ToYCbCr[1, 0] * r + ToYCbCr[1, 1] * g + ToYCbCr[1, 2] * b) + 128
    Cr = (ToYCbCr[2, 0] * r + ToYCbCr[2, 1] * g + ToYCbCr[2, 2] * b) + 128
    new = np.zeros(shape)
    new[:, :, 0] = Y
    new[:, :, 1] = Cb
    new[:, :, 2] = Cr

    show_image("YCbCr img Y", "gray", "gray", Y)
    show_image("YCbCr img Cb", "gray", "gray", Cb)
    show_image("YCbCr img Cr", "gray", "gray", Cr)

    return [Y, Cb, Cr, new.astype(np.uint8)]


def to_RGB(Y, Cb, Cr):
    # converter de yCbCr para RGB
    
    r = ToRGB[0, 0] * Y + ToRGB[0, 1] * (Cb - 128) + ToRGB[0, 2] * (Cr - 128)
    g = ToRGB[1, 0] * Y + ToRGB[1, 1] * (Cb - 128) + ToRGB[1, 2] * (Cr - 128)
    b = ToRGB[2, 0] * Y + ToRGB[2, 1] * (Cb - 128) + ToRGB[2, 2] * (Cr - 128)

    r[r > 255] = 255
    r[r < 0] = 0
    r = np.round(r).astype(np.uint8)

    g[g > 255] = 255
    g[g < 0] = 0
    g = np.round(g).astype(np.uint8)
    b[b > 255] = 255
    b[b < 0] = 0
    b = np.round(b).astype(np.uint8)
    '''
    show_image("YCbCr img Y", "gray", "gray", r)
    show_image("YCbCr img Cb", "gray", "gray", g)
    show_image("YCbCr img Cr", "gray", "gray", b)
    '''
    return [r, g, b]


# ------------------- Exercicio 6 ----------------------------

def sub_sample(downsampling, Y, Cb, Cr):
    if downsampling[2] == 0:
        Cb_d = cv2.resize(Cb, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        Cr_d = cv2.resize(Cr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

    if downsampling[2] == 2:
        Cb_d = cv2.resize(Cb, None, fx=0.5, fy=1, interpolation=cv2.INTER_NEAREST)
        Cr_d = cv2.resize(Cr, None, fx=0.5, fy=1, interpolation=cv2.INTER_NEAREST)

    show_image("Sub_Sampled Cb", "gray", "gray", Cb_d)
    show_image("Sub_Sampled Cr", "gray", "gray", Cr_d)

    return [Y, Cb_d, Cr_d]


def up_sample(downsampling, Y_d, Cb_d, Cr_d):
    if downsampling[2] == 0:
        Cb = cv2.resize(Cb_d, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        Cr = cv2.resize(Cr_d, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

    if downsampling[2] == 2:
        Cb = cv2.resize(Cb_d, None, fx=2, fy=1, interpolation=cv2.INTER_NEAREST)
        Cr = cv2.resize(Cr_d, None, fx=2, fy=1, interpolation=cv2.INTER_NEAREST)
    '''
    show_image("Up_Sampled Cb", "gray", "gray", Cb)
    show_image("Up_Sampled Cr", "gray", "gray", Cr)
    '''
    return [Y_d, Cb, Cr]


# ------------------- Exercicio 7 ----------------------------
def DCT(array):
    Y_dct = dct(dct(array[0], norm='ortho').T, norm='ortho').T
    Cb_dct = dct(dct(array[1], norm='ortho').T, norm='ortho').T
    Cr_dct = dct(dct(array[2], norm='ortho').T, norm='ortho').T

    show_image("Y_dct", "gray", "gray", np.log(np.abs(Y_dct) + 0.0001))
    show_image("Cb_dct", "gray", "gray", np.log(np.abs(Cb_dct) + 0.0001))
    show_image("Cr_dct", "gray", "gray", np.log(np.abs(Cr_dct) + 0.0001))

    return [Y_dct, Cb_dct, Cr_dct]


def IDCT(Y_dct, Cb_dct, Cr_dct):
    Y_idct = idct(idct(Y_dct, norm='ortho').T, norm='ortho').T
    Cb_idct = idct(idct(Cb_dct, norm='ortho').T, norm='ortho').T
    Cr_idct = idct(idct(Cr_dct, norm='ortho').T, norm='ortho').T

    '''
    show_image("Y_idct", "gray", "gray", np.log(np.abs(Y_idct) + 0.0001))
    show_image("Cb_idct", "gray", "gray", np.log(np.abs(Cb_idct) + 0.0001))
    show_image("Cr_idct", "gray", "gray", np.log(np.abs(Cr_idct) + 0.0001))
    '''

    return [Y_idct, Cb_idct, Cr_idct]


def DCT8_64(blocks, array):
    array_dct = []
    for k in range(len(array)):
        new = np.zeros(array[k].shape)
        for i in range(0, len(array[k]), blocks):
            for j in range(0, len(array[k][0]), blocks):
                new[i:i + blocks, j:j + blocks] = dct(dct(array[k][i:i + blocks, j:j + blocks], norm='ortho').T, norm='ortho').T
        array_dct.append(new)

    show_image("Y_dct" + str(blocks), "gray", "gray", np.log(np.abs(array_dct[0]) + 0.0001))
    show_image("Cb_dct" + str(blocks), "gray", "gray", np.log(np.abs(array_dct[1]) + 0.0001))
    show_image("Cr_dct" + str(blocks), "gray", "gray", np.log(np.abs(array_dct[2]) + 0.0001))

    return array_dct


def IDCT8_64(blocks, array):
    array_idct = []
    for k in range(len(array)):
        new = np.zeros(array[k].shape)
        for i in range(0, len(array[k]), blocks):
            for j in range(0, len(array[k][0]), blocks):
                new[i:i + blocks, j:j + blocks] = idct(idct(array[k][i:i + blocks, j:j + blocks], norm='ortho').T, norm='ortho').T
        array_idct.append(new)

    '''
    show_image("Y_idct" + str(blocks), "gray", "gray", np.log(np.abs(array_idct[0]) + 0.0001))
    show_image("Cb_idct" + str(blocks), "gray", "gray", np.log(np.abs(array_idct[1]) + 0.0001))
    show_image("Cr_idct" + str(blocks), "gray", "gray", np.log(np.abs(array_idct[2]) + 0.0001))
    '''

    return array_idct


# ------------------- Exercicio 8 ----------------------------


def quant_quality(quality):
    if quality >= 50:
        sf = (100 - quality) / 50
    else:
        sf = 50 / quality

    if sf != 0:
        quality_y = np.round(Q_Y * sf)
        quality_cbcr = np.round(Q_CbCr * sf)
    else:
        quality_y = np.ones(Q_Y.shape, dtype=np.uint8)
        quality_cbcr = np.ones(Q_CbCr.shape, dtype=np.uint8)

    quality_y[quality_y > 255] = 255
    quality_y[quality_y < 1] = 1
    quality_y = quality_y.astype(np.uint8)
    quality_cbcr[quality_cbcr > 255] = 255
    quality_cbcr[quality_cbcr < 1] = 1
    quality_cbcr = quality_cbcr.astype(np.uint8)
    return quality_y, quality_cbcr


def quantization(array, quality_y, quality_cbcr, block):
    array_quant = []
    for k in range(3):
        if k == 0:
            quality = quality_y
        else: 
            quality = quality_cbcr

        new = np.empty(array[k].shape)
        lenght = array[k].shape
        for i in range(0, lenght[0], block):
            for j in range(0, lenght[1], block):
                new[i:i + block, j:j + block] = (array[k][i:i + block, j:j + block]) / quality
        new = np.round(new).astype(int)
        array_quant.append(new)

    #print("Y_Q -> ", array_quant[0][8:16, 8:16])

    show_image("Quantization Y", "gray", "gray", np.log(np.abs(array_quant[0]) + 0.0001))
    show_image("Quantization Cb", "gray", "gray", np.log(np.abs(array_quant[1]) + 0.0001))
    show_image("Quantization Cr", "gray", "gray", np.log(np.abs(array_quant[2]) + 0.0001))

    return array_quant

def inv_quantization(array, quality_y, quality_cbcr, block):
    array_invQuant = []
    for k in range(3):
        if k == 0:
            quality = quality_y
        else: 
            quality = quality_cbcr

        new = np.zeros(array[k].shape)
        lenght = array[k].shape
        for i in range(0, lenght[0], block):
            for j in range(0, lenght[1], block):
                new[i:i + block, j:j + block] = (array[k][i:i + block, j:j + block]) * quality
       
        array_invQuant.append(new)

    '''
    show_image("Invert Quantization Y", "gray", "gray", np.log(np.abs(array_invQuant[0]) + 0.0001))
    show_image("Invert Quantization Cb", "gray", "gray", np.log(np.abs(array_invQuant[1]) + 0.0001))
    show_image("Invert Quantization Cr", "gray", "gray", np.log(np.abs(array_invQuant[2]) + 0.0001))
    '''
    return array_invQuant


# ------------------- Exercicio 9 -----------------------------

def DPCM(array, blocks):
    dpcm = []
    for i in range (3):
        diff = array[i].copy()
        for k in range(0, len(array[i]), blocks):
            for j in range(0, len(array[i][0]), blocks):
                if j == 0:
                    if k != 0:
                        diff[k][j] = array[i][k][j] - array[i][k-blocks][len(array[i][0])-blocks-1]
                else:
                    diff[k][j] = array[i][k][j] - array[i][k][j-blocks]
        dpcm.append(diff)

    print("Y_DPCM -> ", dpcm[0][8:16, 8:16])
    show_image("DPCM Y", "gray", "gray", np.log(np.abs(dpcm[0]) + 0.0001))
    show_image("DPCM Cb", "gray", "gray", np.log(np.abs(dpcm[1]) + 0.0001))
    show_image("DPCM Cr", "gray", "gray", np.log(np.abs(dpcm[2]) + 0.0001))

    return dpcm

def IDPCM(array, blocks):
    idpcm = []
    for k in range (3):
        dc = array[k].copy()
        for i in range(0, len(array[k]), blocks):
            for j in range(0, len(array[k][0]), blocks):
                if j == 0:
                    if i != 0:
                        dc[i][j] = dc[i - blocks][len(array[k][0])-blocks-1] + array[k][i][j]
                else:
                    dc[i][j] = dc[i][j-blocks] + array[k][i][j]

        idpcm.append(dc)

    #print("Y_IDPCM -> ", idpcm[0][8:16, 8:16])
    '''
    show_image("IDPCM Y", "gray", "gray", np.log(np.abs(idpcm[0]) + 0.0001))
    show_image("IDPCM Cb", "gray", "gray", np.log(np.abs(idpcm[1]) + 0.0001))
    show_image("IDPCM Cr", "gray", "gray", np.log(np.abs(idpcm[2]) + 0.0001))
    '''
    return idpcm 

# ------------------- Exercicio 10 ----------------------------
def MSE(img_original, img_rec):
    mse = np.sum((img_original.astype(float) - img_rec.astype(float)) ** 2)
    mse /= float(img_original.shape[0] * img_original.shape[1])
    return mse


def RMSE(mse):
    rmse = math.sqrt(mse)
    return rmse


def SNR(img_original, mse):
    P = np.sum(img_original.astype(float) ** 2)
    P /= float(img_original.shape[0] * img_original.shape[1])
    snr = 10 * math.log10(P/mse)
    return snr


def PSNR(mse, img_original):
    original = img_original.astype(float)
    max_ = np.max(original) ** 2
    psnr = 10 * math.log10(max_/mse)
    return psnr



# ===== FUNCOES PRINCIPAIS =====

def encoder(img):
    print("==================\nEncoder\n==================\n")
    pad = padding(img)
    split = split_channel(pad)
    array_YCbCr = to_YCbCr(split[0], split[1], split[2], split[3])
    sub = sub_sample((4, 2, 2), array_YCbCr[0], array_YCbCr[1], array_YCbCr[2])
    DCT(sub)
    array_dct8 = DCT8_64(8, sub)
    #DCT8_64(64, array_YCbCr[0], sub[1], sub[2])
    Y_quality, CbCr_quality = quant_quality(qf)
    quant = quantization(array_dct8, Y_quality, CbCr_quality, 8)
    dpcm = DPCM(quant, 8)
    dpcm.append(Y_quality)
    dpcm.append(CbCr_quality)
    return [dpcm, array_YCbCr[0]]


def decoder(array, img):
    print("==================\nDecoder\n==================\n")
    idpcm = IDPCM(array, 8)
    inv = inv_quantization(idpcm, array[3], array[4], 8)
    array_idct = IDCT8_64(8, inv)
    upsam = up_sample((4, 2, 2), array_idct[0], array_idct[1], array_idct[2])
    Y_d = upsam[0]
    rgb = to_RGB(upsam[0], upsam[1], upsam[2])
    img_rev = reverse_split(rgb)
    img_final = reverse_padding(img_rev, img)
    return img_final, Y_d


def statistics(img_original, img_rec, name):
    print("===============================\n" + str(qf) + " Quality factor for " +  name + "\n===============================\n")
    mse = MSE(img_original, img_rec)
    print("MSE: " + str(mse))
    rmse = RMSE(mse)
    print("RMSE: " + str(rmse))
    snr = SNR(img_original, mse)
    print("SNR: " + str(snr))
    psnr = PSNR(mse, img_original)
    print("PSNR: " + str(psnr))


def main():

    file_name = 'barn_mountains'
    img = plt.imread("imagens/" + file_name + ".bmp")
    print("Tamanhos:")
    compress_3ways(os.path.join(os.path.dirname(__file__), "imagens/"), "barn_mountains")
    compress_3ways(os.path.join(os.path.dirname(__file__), "imagens/"), "logo")
    compress_3ways(os.path.join(os.path.dirname(__file__), "imagens/"), "peppers")
    show_image(file_name, "gray", "None", img)
   
    
    img_encoder, Y_e = encoder(img)
    img_decoder, Y_d = decoder(img_encoder, img)

    E = np.absolute(Y_e - Y_d)
    show_image("Y error", "gray", "gray", E)
    
    #print(np.mean(E))
    
    statistics(img, img_decoder, file_name)
    

if __name__ == "__main__":
    main()