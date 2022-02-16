# MSInformalED dataset

`MSInformalED.zip` file includes `train.json`, `dev.json` and `test.json` of MSInformalED dataset. The data format of the three `.json` files are the same. Take an instance in `train.json` as an example:

```JSON5
[
    {
        "id": 90076777,
        "content": [
            "客服:Hi~请问您是遇到以下问题吗？,",
            "客服:您好，欢迎您进入人工客服，请输入您的问题，人工客服小美会竭诚为您服务的~,",
            "用户:餐内异物,",
            "客服:您好，我是本次为您服务的客服小美_94372，很高兴为您服务～,",
            "客服:亲亲是吃到异物了吗,",
            "用户:对,",
            "客服:十分抱歉了给您添麻烦了这边是美团外卖这边给您转接到食安专线可以嘛,",
            "用户:好的,",
            "客服:您好，我是本次为您服务的客服小美_179838，很高兴为您服务～,",
            "客服:您好，我是美团客服，很高兴为您服务。查看到您名下之前反馈过【普通异物-[鸡蛋壳][头发][肉上有毛]等】的问题，请问您是否咨询该问题的处理进展？,",
            "用户:不是,",
            "用户:是这个订单,",
            "客服:亲亲，请问订单是有什么问题呢[可怜][鲜花]，辛苦您详细描述一下呢,",
            "客服:亲亲，小美等的好辛苦[可怜]，由于您在3分钟内没有任何回应，系统已经自动结束对话了，小美不得不与您分开了呢>_<。如仍需帮助可再次发起会话，如没有其他问题，麻烦对小美的服务进行评价哦，满意请给朵鲜花，不满意请给鸡蛋，谢谢您~[亲亲]"
        ],
        "events": [
            {
                "type": "异物",
                "trigger_word": "异物",
                "sent_id": 2,
                "offset": [
                    5,
                    7
                ]
            },
            {
                "type": "异物",
                "trigger_word": "异物",
                "sent_id": 4,
                "offset": [
                    8,
                    10
                ]
            },
            {
                "type": "异物",
                "trigger_word": "异物",
                "sent_id": 9,
                "offset": [
                    35,
                    37
                ]
            },
            {
                "type": "异物",
                "trigger_word": "鸡蛋壳",
                "sent_id": 9,
                "offset": [
                    39,
                    42
                ]
            },
            {
                "type": "异物",
                "trigger_word": "头发",
                "sent_id": 9,
                "offset": [
                    44,
                    46
                ]
            },
            {
                "type": "异物",
                "trigger_word": "毛",
                "sent_id": 9,
                "offset": [
                    51,
                    52
                ]
            }
        ],
        "domain": "Text Conversations"
    }
]
```
