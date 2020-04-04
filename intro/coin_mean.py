import random

def coin_toss_mean(n=100):
    total = 0
    for i in range(n + 1):
        total += random.randrange(0, 2, 1)

    print(f"{int(total / n * 100)}%")


def coin_toss_continuous_mean(n=200):
    total = 0
    mn = 0
    for i in range(n + 1):
        total += random.randrange(0, 2, 1)

        mn = (1 - 1.0 / n) * mn + 1.0/ n *  random.randrange(0, 2, 1)
        print(mn)


coin_toss_continuous_mean()