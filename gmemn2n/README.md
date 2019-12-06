# Gated End2End Memory Network

# bAbI 1k Performance

The results are affected by the initialization a lot. The following results are got by setting the ``random_state`` to 2033, epochs 200.

```
$ python main.py --use_cuda
```

| task | train acc | val acc |
| ---- | --------- | ------- |
| 1    | 1.0000    |   |
| 2    | 0.9390    |   |
| 3    | 0.8890    |   |
| 4    | 0.9990    |   |
| 5    | 0.8930    |   |
| 6    | 0.9980    |   |
| 7    | 0.9980    |   |
| 8    | 0.9030    |   |
| 9    | 0.9960    |   |
| 10   | 0.8970    |   |
| 11   | 1.0000    |   |
| 12   | 1.0000    |   |
| 13   | 0.9800    |   |
| 14   | 1.0000    |   |
| 15   | 0.7440    |   |
| 16   | 0.7440    |   |
| 17   | 0.7680    |   |
| 18   | 0.9660    |   |
| 19   | 0.3680    |   |
| 20   | 1.0000    |   |

# Reference

* <https://mp.weixin.qq.com/s/5UrKnpkA1mAFYKOeavtn2g>
