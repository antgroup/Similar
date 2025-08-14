import os
import json
import shutil

filepath = "generate_data/tasks/vwa/test_reddit"
shopping_task_ids = [21, 22, 23, 24, 25, 26, 47, 48, 49, 50, 51, 96, 117, 118, 124, 125, 126, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 188, 189, 190, 191, 192, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 238, 239, 240, 241, 242, 260, 261, 262, 263, 264, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 298, 299, 300, 301, 302, 313, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 351, 352, 353, 354, 355, 358, 359, 360, 361, 362, 368, 376, 384, 385, 386, 387, 388, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 465, 466, 467, 468, 469, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 528, 529, 530, 531, 532, 571, 572, 573, 574, 575, 585, 586, 587, 588, 589, 653, 654, 655, 656, 657, 689, 690, 691, 692, 693, 792, 793, 794, 795, 796, 797, 798]
shopping_admin_task_ids = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 41, 42, 43, 62, 63, 64, 65, 77, 78, 79, 94, 95, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 119, 120, 121, 122, 123, 127, 128, 129, 130, 131, 157, 183, 184, 185, 186, 187, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 243, 244, 245, 246, 247, 288, 289, 290, 291, 292, 344, 345, 346, 347, 348, 374, 375, 423, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 470, 471, 472, 473, 474, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 676, 677, 678, 679, 680, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 790]
gitlab_task_ids = [44, 45, 46, 102, 103, 104, 105, 106, 132, 133, 134, 135, 136, 156, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 205, 206, 207, 258, 259, 293, 294, 295, 296, 297, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 314, 315, 316, 317, 318, 339, 340, 341, 342, 343, 349, 350, 357, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 522, 523, 524, 525, 526, 527, 533, 534, 535, 536, 537, 567, 568, 569, 570, 576, 577, 578, 579, 590, 591, 592, 593, 594, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 736, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 783, 784, 785, 786, 787, 788, 789, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811]
reddit_task_ids = [27, 28, 29, 30, 31, 66, 67, 68, 69, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 580, 581, 582, 583, 584, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735]

done = []
to_do = []
without_all_length = []

for filedir in os.listdir(filepath):
    if not os.path.isdir(os.path.join(filepath, filedir)):
        continue

    # print("filedir = ", filedir)
    if filedir.isdigit():
        print("\nnow = ", filedir)
        action_list_file = os.path.join(filepath, filedir, "action_list.json")

        if not os.path.exists(action_list_file):
            to_do.append(int(filedir))
            # shutil.rmtree(os.path.join(filepath, filedir))
        else:
            with open(action_list_file, 'r', encoding='UTF-8') as f:
                action_list = json.load(f)

            all_length_file = os.path.join(filepath, filedir, "all_length.text")

            # if not os.path.exists(all_length_file):
            #     all_length = 0.0
            #     with open(all_length_file, 'w') as f:
            #         f.write(str(all_length))

            if not os.path.exists(all_length_file):
                without_all_length.append(int(filedir))
                to_do.append(int(filedir))
                continue

            with open(all_length_file, "r", encoding='utf-8') as f:
                all_length = float(f.readline())

            if all_length == 0.0:
                print("\n000000")
                done.append(int(filedir))
            else:
                print("\n111111")
                evaluation_score_file = os.path.join(filepath, filedir, "evaluation_score.json")
                if os.path.exists(evaluation_score_file):
                    # done.append(int(filedir))

                    with open(evaluation_score_file, 'r', encoding='UTF-8') as f:
                        evaluation_score_dict = json.load(f)

                    flag = 0
                    for (key, value) in evaluation_score_dict.items():
                        if ('stop' in value['action']):
                            print("\nthis task is done.")
                            done.append(int(filedir))
                            flag = 1
                            break
                    if flag == 0:
                        print('\nthis task is not be fully completed.')
                        to_do.append(int(filedir))
                else:
                    to_do.append(int(filedir))


done.sort()
to_do.sort()
without_all_length.sort()
# rest = list(set(gitlab_task_ids) - set(done))
# rest.sort()

print("\ndone = ", done)
print("num done = %d\n" % len(done))

print("\nto_do = ", to_do)
print("num to do = %d\n" % len(to_do))

print("\nwithout_all_length = ", without_all_length)
print("num without_all_length = %d\n" % len(without_all_length))
# print("\nrest task = ", rest)


