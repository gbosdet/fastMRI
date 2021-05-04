import torch
import h5py
import pathlib
from fastmri.data.mri_data import fetch_dir
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

def train():
    sns.set()
    path_config = pathlib.Path("../../fastmri_dirs.yaml")
    data_path = fetch_dir("knee_path", path_config)
    training_path = data_path / "singlecoil_train"
    validation_path = data_path / "singlecoil_val"
    likely_rows = [320,321,319,322,318,317,323,316,315,314,313,324,312,326,329,325,327,330,311,310,328,309,331,332,308,
                   306,307,333,341,340,338,305,298,335,302,296,299,334,303,347,337,297,304,336,348,344,339,279,346,288,
287,294,360,300,359,342,349,343,283,291,295,361,293,276,289,363,292,286,357,285,281,277,364,284,290,280,350,282,353,352,
301,358,345,355,362,354,351,275,356,278]

    likely_tuples =[(320, 25012),
(321, 25012),
(319, 25012),
(322, 25012),
(318, 25012),
(317, 25012),
(323, 25012),
(316, 25012),
(315, 25012),
(314, 25012),
(313, 25012),
(324, 25012),
(312, 25012),
(326, 25012),
(329, 25011),
(325, 25012),
(327, 25012),
(330, 25011),
(311, 25012),
(310, 25012),
(328, 25012),
(309, 25012),
(331, 25008),
(332, 25007),
(308, 25010),
(306, 25005),
(307, 25004),
(333, 25002),
(341, 24885),
(340, 24922),
(338, 24962),
(305, 25002),
(298, 24879),
(335, 24997),
(302, 24973),
(296, 24817),
(299, 24923),
(334, 25000),
(303, 24986),
(347, 24599),
(337, 24981),
(297, 24857),
(304, 25001),
(336, 24979),
(348, 24551),
(344, 24739),
(339, 24945),
(279, 21970),
(346, 24651),
(288, 24244),
(287, 24141),
(294, 24710),
(360, 22195),
(300, 24956),
(359, 22615),
(342, 24838),
(349, 24442),
(343, 24810),
(283, 23386),
(291, 24527),
(295, 24774),
(361, 21816),
(293, 24667),
(276, 20030),
(289, 24343),
(363, 20411),
(292, 24628),
(286, 24006),
(357, 23271),
(285, 23855),
(365, 18608),
(281, 22786),
(277, 20849),
(364, 19517),
(284, 23682),
(290, 24444),
(280, 22454),
(350, 24349),
(282, 23177),
(353, 23974),
(352, 24182),
(301, 24980),
(358, 22898),
(367, 15966),
(345, 24700),
(273, 16419),
(355, 23717),
(362, 21099),
(354, 23898),
(351, 24239),
(275, 19002),
(274, 17715),
(366, 17344),
(356, 23384),
(278, 21526)]

    top162 = [(320, 25012),
(321, 25012),
(319, 25012),
(322, 25012),
(318, 25012),
(317, 25012),
(323, 25012),
(316, 25012),
(315, 25012),
(314, 25012),
(313, 25012),
(324, 25012),
(312, 25012),
(326, 25012),
(329, 25012),
(325, 25012),
(327, 25012),
(330, 25011),
(311, 25012),
(310, 25012),
(328, 25012),
(309, 25012),
(331, 25010),
(332, 25010),
(308, 25012),
(306, 25008),
(307, 25006),
(333, 25007),
(341, 24954),
(340, 24971),
(338, 24993),
(305, 25005),
(298, 24951),
(335, 25004),
(302, 24989),
(296, 24931),
(299, 24972),
(334, 25004),
(303, 25001),
(347, 24837),
(337, 24998),
(297, 24941),
(304, 25004),
(336, 24996),
(348, 24800),
(344, 24887),
(339, 24979),
(279, 24177),
(346, 24852),
(288, 24682),
(287, 24669),
(294, 24882),
(360, 24228),
(300, 24989),
(359, 24288),
(342, 24926),
(349, 24741),
(343, 24931),
(283, 24481),
(291, 24793),
(295, 24904),
(361, 24118),
(293, 24861),
(276, 24017),
(289, 24712),
(363, 23934),
(292, 24843),
(370, 23268),
(286, 24611),
(357, 24427),
(285, 24570),
(365, 23698),
(281, 24333),
(277, 24072),
(364, 23886),
(284, 24514),
(290, 24764),
(280, 24273),
(350, 24725),
(282, 24401),
(353, 24578),
(352, 24663),
(301, 24995),
(358, 24334),
(267, 22896),
(367, 23534),
(275, 23858),
(351, 24672),
(355, 24499),
(373, 22818),
(369, 23310),
(354, 24574),
(345, 24876),
(368, 23423),
(272, 23608),
(356, 24414),
(269, 23200),
(278, 24137),
(265, 22584),
(366, 23677),
(270, 23346),
(274, 23784),
(374, 22627),
(372, 22872),
(273, 23689),
(362, 24021),
(271, 23454),
(371, 23092),
(266, 22801),
(268, 23065)]
    # top162.sort()
    # print(top162)
    # print(len(top162))
    model = nn.Sequential(nn.Linear(640, 2048), nn.ReLU(), nn.Linear(2048, 640), nn.Softmax())
    optimizer = torch.optim.Adam(model.parameters())
    counter = np.zeros(640, dtype=np.int)
    for epoch in range(1):
        count = 1
        for path in training_path.iterdir():
            file = h5py.File(path, mode="r")
            # print(list(file.keys()))
            kspace = file['kspace']
            print(count)
            count += 1
            for i in range(5, kspace.shape[0]-5):
                slice = torch.from_numpy(np.array(kspace[i]))
                row_sums = torch.sum(torch.sqrt(slice.real ** 2 + slice.imag ** 2), 1)
                top_sums = torch.topk(row_sums, 162)[1]
                for sum in top_sums:
                    m = int(sum)
                    counter[m] +=1
                # print("Slice Num:", i, ", Max row:", m)

    # for key, val in counter.items():
    #     if val > 22500:
    #         print("(" + str(key) + ", " + str(val) + "), ", sep="")
    plt.title("Top 162 Row Sum vs Row Index")
    plt.xlabel("Row Index")
    plt.ylabel("Times in top 162")
    plt.plot(list(range(640)), counter)
    plt.savefig("Top162x.png")
    plt.close()



def top10Loss(output, target):
    #scaling_factor = 0.05
    return torch.max(target)[0] - output*target





if __name__ == "__main__":
    train()