import json
import logging

from layotto.client.base import AntLayottoClient
from layotto.client.request.inference import (
    Debug,
    InferenceRequest,
    Item,
    LayottoInferenceConfig,
    MayaConfig,
    TensorFeatures,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


def test_maya_stream():
    # 1. 需要初始化Mosn,这一步是对Mosn进行了初始化，这个是全局唯一的，即全局只需要应用启动时执行一次即可!!
    client = AntLayottoClient()
    client.initialize_app("iseecore")
    # 等Mosn2.19发布后跑这个ut
    # 2. 这一步是初始化Layotto的maya能力，全局执行一次即可
    client.init_inference(LayottoInferenceConfig("iseecore", "maya"))
    item = Item()
    item.set_item_id("123456789")
    tensor_features = TensorFeatures()
    tensor_features.set_string_values(
        [
            json.dumps(
                {
                    "__entry_point__": "openai.chat.completion",
                    "model": "auto",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "hi"},
                    ],
                    "stream": False,
                    # 以下5个推理参数会影响推理速度和效果，如非必要可不填
                    "max_tokens": 512,
                    "temperature": 0.4,
                    "repetition_penalty": 1.09,
                    "top_p": 0.95,
                    "top_k": 20,
                }
            )
        ]
    )
    item.set_tensor_features({"data": tensor_features})
    config = MayaConfig()

    # 请求超时时间，非必填，默认600ms
    config.request_time_out = 100000

    request = InferenceRequest(
        scene_name="DeepSeek_R1_Test_Agent_214",
        chain_name="DeepSeek_R1_Test_Agent_214_v1",
        items=[item],
        config=config,
        debug=Debug.OPEN,
    )
    # 通过for不断读取stream的返回，不支持async
    for resp in client.stream_inference(request):
        logger.info(
            f"maya stream call success with resp, "
            f"object_attributes: {resp.items[0].object_attributes}, "
            f"attributes: {resp.items[0].attributes}, "
            f"item_id: {resp.items[0].item_id}, "
            f"score: {resp.items[0].score}, "
            f"scores: {resp.items[0].scores}, "
            f"servers: {resp.servers}, "
            f"rt: {resp.rt}, "
            f"success: {resp.success} "
        )

if __name__ == "__main__":
    test_maya_stream()