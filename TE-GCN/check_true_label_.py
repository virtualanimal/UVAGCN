import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import logging

class_names = {
    "A000": "drink",
    "A001": "eat snacks",
    "A002": "brush hair",
    "A003": "drop something",
    "A004": "pick up something",
    "A005": "throw away something",
    "A006": "sit down",
    "A007": "stand up",
    "A008": "applaud",
    "A009": "read",
    "A010": "write",
    "A011": "put on a coat",
    "A012": "take off a coat",
    "A013": "put on glasses",
    "A014": "take off glasses",
    "A015": "put on a hat",
    "A016": "take off a hat",
    "A017": "throw away a hat",
    "A018": "cheer",
    "A019": "wave hands",
    "A020": "kick something",
    "A021": "reach into pockets",
    "A022": "jump on single leg",
    "A023": "jump on two legs",
    "A024": "make a phone call",
    "A025": "play with cell phones",
    "A026": "point somewhere",
    "A027": "look at the watch",
    "A028": "rub hands",
    "A029": "bow",
    "A030": "shake head",
    "A031": "salute",
    "A032": "cross palms together",
    "A033": "cross arms in front to say no",
    "A034": "wear headphones",
    "A035": "take off headphones",
    "A036": "make a shh sign",
    "A037": "touch the hair",
    "A038": "thumb up",
    "A039": "thumb down",
    "A040": "make an OK sign",
    "A041": "make a victory sign",
    "A042": "punch with fists",
    "A043": "figure snap",
    "A044": "open the bottle",
    "A045": "smell",
    "A046": "squat",
    "A047": "apply cream to face",
    "A048": "apply cream to hands",
    "A049": "grasp a bag",
    "A050": "put down a bag",
    "A051": "put something into a bag",
    "A052": "take something out of a bag",
    "A053": "open a box",
    "A054": "move a box",
    "A055": "put up hands",
    "A056": "put hands on hips",
    "A057": "wrap arms around",
    "A058": "shake arms",
    "A059": "step on the spot walk",
    "A060": "kick aside",
    "A061": "kick backward",
    "A062": "cough",
    "A063": "sneeze",
    "A064": "yawn",
    "A065": "blow nose",
    "A066": "stagger",
    "A067": "headache",
    "A068": "chest discomfort",
    "A069": "backache",
    "A070": "neck-ache",
    "A071": "vomit",
    "A072": "use a fan",
    "A073": "stretch body",
    "A074": "punching someone",
    "A075": "kicking someone",
    "A076": "pushing someone",
    "A077": "slap someone on the back",
    "A078": "point someone",
    "A079": "hug",
    "A080": "give something to someone",
    "A081": "steal something from other’s pocket",
    "A082": "rob something from someone",
    "A083": "shake hands",
    "A084": "walk toward someone",
    "A085": "walk away from someone",
    "A086": "hit someone with something",
    "A087": "threat some with a knife",
    "A088": "bump into someone",
    "A089": "walk side by side",
    "A090": "high five",
    "A091": "drink a toast",
    "A092": "move something with someone",
    "A093": "take a phone for someone",
    "A094": "stalk someone",
    "A095": "whisper in someone’s ear",
    "A096": "exchange something with someone",
    "A097": "lend an arm to support someone",
    "A098": "rock-paper-scissors",
    "A099": "hover",
    "A100": "land",
    "A101": "land at designated locations",
    "A102": "move forward",
    "A103": "move backward",
    "A104": "move left",
    "A105": "move right",
    "A106": "ascend",
    "A107": "descend",
    "A108": "accelerate",
    "A109": "decelerate",
    "A110": "come over here",
    "A111": "stay where you are",
    "A112": "rear right turn",
    "A113": "rear left turn",
    "A114": "abandon landing",
    "A115": "all clear",
    "A116": "not clear",
    "A117": "have command",
    "A118": "follow me",
    "A119": "turn left",
    "A120": "turn right",
    "A121": "throw litter",
    "A122": "dig a hole",
    "A123": "mow",
    "A124": "set on fire",
    "A125": "smoke",
    "A126": "cut the tree",
    "A127": "fishing",
    "A128": "pick a lock",
    "A129": "pollute walls",
    "A130": "hold someone hostage",
    "A131": "threat someone with a gun",
    "A132": "wave a goodbye",
    "A133": "chase someone",
    "A134": "comfort someone",
    "A135": "drag someone",
    "A136": "sweep the floor",
    "A137": "mop the floor",
    "A138": "bounce the ball",
    "A139": "shoot at the basket",
    "A140": "swing the racket",
    "A141": "leg pressing",
    "A142": "escape (to survive)",
    "A143": "call for help",
    "A144": "wear a mask",
    "A145": "take off a mask",
    "A146": "bend arms around someone’s shoulder",
    "A147": "run",
    "A148": "stab someone with a knife",
    "A149": "throw a frisbee",
    "A150": "carry a carrying pole",
    "A151": "use a lever to lift something",
    "A152": "walk",
    "A153": "open an umbrella",
    "A154": "close an umbrella"
}


two_person_action = {    #32
    "A074": "punching someone",
    "A075": "kicking someone",
    "A076": "pushing someone",
    "A077": "slap someone on the back",
    "A078": "point someone",
    "A079": "hug",
    "A080": "give something to someone",
    "A081": "steal something from other’s pocket",
    "A082": "rob something from someone",
    "A083": "shake hands",
    "A084": "walk toward someone",
    "A085": "walk away from someone",
    "A086": "hit someone with something",
    "A087": "threat some with a knife",
    "A088": "bump into someone",
    "A089": "walk side by side",
    "A090": "high five",
    "A091": "drink a toast",
    "A092": "move something with someone",
    "A093": "take a phone for someone",
    "A094": "stalk someone",
    "A095": "whisper in someone’s ear",
    "A096": "exchange something with someone",
    "A097": "lend an arm to support someone",
    "A130": "hold someone hostage",
    "A131": "threat someone with a gun",
    "A132": "wave a goodbye",
    "A133": "chase someone",
    "A134": "comfort someone",
    "A135": "drag someone",
    "A146": "bend arms around someone’s shoulder",
    "A148": "stab someone with a knife",
}

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='model_cls_result.log',
                    filemode='w')

# 记录每个类别的正确和错误数量
correct_class_count = {}
incorrect_class_count = {}

# pre_label_path =  '/data/lyp/Skeleton_Based_Action_Recognition/TE-GCN/test_train/epoch1_test_score.pkl'
# label_path = '/data/lyp/Skeleton_Based_Action_Recognition/resources/data/train_label.npy'
# train_data_path = '/data/lyp/Skeleton_Based_Action_Recognition/resources/data/train_joint.npy'

pre_label_path =  '/data/lyp/Skeleton_Based_Action_Recognition/TE-GCN/tegcn_8_modality_ensemble_testA.npy'
label_path = '/data/lyp/Skeleton_Based_Action_Recognition/resources/data/val_label.npy'

labels = np.load(label_path)
preds = np.load(pre_label_path)

# with open(pre_label_path,'rb') as f:
#     preds = list(pickle.load(f).items())

CR =0

for i in tqdm(range(len(labels))):
    label = labels[i]
    pred = preds[i]
    pred = np.argmax(pred)

    if label == pred:
        CR += 1

        if label in correct_class_count:
            correct_class_count[label] += 1
        else:
            correct_class_count[label] = 1
    else:

        if label in incorrect_class_count:
            incorrect_class_count[label] += 1
        else:
            incorrect_class_count[label] = 1

# 对正确和错误分类计数按数量排序
sorted_correct_classes = sorted(correct_class_count.items(), key=lambda x: x[1], reverse=True)
sorted_incorrect_classes = sorted(incorrect_class_count.items(), key=lambda x: x[1], reverse=True)

cr_motion = 0
error_motion = 0

# 使用类别名称记录日志
logging.info("Correctly Classified Categories (sorted by count):")
for label, count in sorted_correct_classes:
    class_name = class_names.get(f"A{label:03d}", "Unknown")
    logging.info(f"Class {class_name} (A{label:03d}): {count}")
    if f"A{label:03d}" in two_person_action.keys():
        cr_motion += count


logging.info("Incorrectly Classified Categories (sorted by count):")
for label, count in sorted_incorrect_classes:
    class_name = class_names.get(f"A{label:03d}", "Unknown")
    logging.info(f"Class {class_name} (A{label:03d}): {count}")

    if f"A{label:03d}" in two_person_action.keys():
        error_motion += count

print(f"正确分类动作数量：{cr_motion}")

print(f"错误分类动作数量：{error_motion}")

