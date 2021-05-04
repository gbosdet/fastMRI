# from tensorflow.python.summary.summary_iterator import summary_iterator
#
# for e in summary_iterator("D:\Computer Science\Deep Learning\\fastMRI\\fastmri_examples\\varnet\\varnet\\varnet_demo\lightning_logs\\version_3\events.out.tfevents.1619273380.760d471526ee.399.0"):
#     for v in e.summary.value:
#         if v.tag == "val_metrics/ssim":
#             print(str(v.simple_value) +", ")
#
# print("random")
# for e in summary_iterator("D:\Computer Science\Deep Learning\\fastMRI\\fastmri_examples\\varnet\\varnet\\varnet_demo\lightning_logs\\version_4\events.out.tfevents.1619367795.cc8f1ffe286f.402.0"):
#     for v in e.summary.value:
#         if v.tag == "val_metrics/ssim":
#             print(str(v.simple_value) + ", ")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()



epochs8 = [1, 2, 3, 4, 5, 6, 7, 8]
epochs5 = [1, 2, 3, 4, 5]
epochs6 = [1, 2, 3, 4, 5, 6]
epochs10 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
epochs13 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

random = [0.7257665991783142,
0.7331594228744507,
0.7357654571533203,
0.7368056178092957,
0.7382997870445251,
0.738771378993988,
0.7395404577255249,
0.7396954894065857]
random_real = [0.7258, 0.7332, 0.7348, 0.7356, 0.7371, 0.7396, 0.7398, 0.74]
equispaced = [0.7263679504394531,
0.732336699962616,
0.735085129737854,
0.7359210252761841,
0.7367194890975952,
0.7392144203186035,
0.7393988370895386,
0.7395454049110413]
maxrows = [0.7645, 0.7657, 0.7675, 0.7679, 0.7684, 0.7685, 0.7687, 0.7688,0.7688,0.7689]
middlerows = [0.754, 0.7548, 0.7556, 0.7558, 0.7562, 0.7568, 0.7569, 0.757, 0.757, 0.757]
offset_equispaced = [0.7258, 0.7303, 0.7344, 0.7356, 0.7361]
trained_model = [0.7555, 0.756, 0.7564, 0.7566, 0.7573, 0.7574]

plt.plot(epochs8, random_real)
plt.plot(epochs8, equispaced)
plt.plot(epochs5, offset_equispaced)
plt.plot(epochs10, maxrows)
plt.plot(epochs10, middlerows)
plt.plot(epochs6, trained_model)

plt.legend(["random", "equispacing", "offset equispaced", "maximum rows", "central rows", "DNN choice"])
plt.ylabel("SSIM")
plt.xlabel("Epoch")
plt.title("SSIM Score by Epoch")
plt.tight_layout()
plt.savefig("MaskingSSIM")
plt.close()