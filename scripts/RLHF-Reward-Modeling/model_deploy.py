import uuid

from max_common.enums.ModelAssetTypeEnum import ModelTypeEnum
from max_common.model.api.model_asset import ModelAssetApi
from max_common.model.param.model_asset_param import (
RegisterModelParams,
)

# 用户信息, modelhub 模型有鉴权逻辑
user_name = "miaobingchen.mbc"
user_id = "456420"

tenant_id = "e7769abb" # 此 id 为体验租户，如果不知道默认填写这个
# 如果需要调整查询 max 平台上的租户 id：https://amax.alipay.com/maas/modelhub?tenantId=e7769abb

# 模型名称，模型名称重复，会导致创建git|zeta仓库失败，可以加一些特殊标识，比如uuid
model_name = "miaobingchen_Similar_Llava_8B_v01"
info = {
    "user_name": user_name,
    "user_id": user_id,
    "tenant_id": tenant_id,
    "local_new_model_asset_folder_path": "/mnt/prev_nas/virtual_agent/MBC/RLHF-Reward-Modeling/checkpoints/similar/similar_llava", # 填写存放模型文件的本地文件夹
    "model_name": model_name,
    "description": "register model", # modelhub 上模型的表述信息
    "git_project_name": model_name, # 注册到modelhub的模型，最终存储在了 zeta｜git 仓库, 这里是仓库的名字
    "base_only": True, # 如果是自己上传的基座，或者非微调模型，这里设置为True，如果是微调模型(e.g. Lora)，需要关联对应的基座，设置为False
    #"base_model_id": "xxxx", # 如果是Lora模型，在平台上会有根据 基座模型筛选的逻辑，需要填写 基座模型的ID，e.g. 在ais上找到基座模型详情页可以看到 https://aistudio.alipay.com/project/marketplace/model/detail/99739
    "release_tag": "test_tag",  # zeta|git仓库 可以记录tag信息
    "release_desc": "test_desc",
    "dfs_deploy_info": False, # 不用改，固定false
}
# 通过dict 创建 dataclass
resiter_param = RegisterModelParams(**info)
print(ModelAssetApi.register_modelhub_asset(resiter_param))

# 最后会输出 modelhub_id，可以 拼接链接 https://aistudio.alipay.com/project/marketplace/model/detail/${modelhub_id}，在ais上找到，在推理组件面板可选