def cost_balancing(w1: list, w2: list, cost):
    """
    Suppose you have 60% equity, 40% fixed income and you want to make the weight be 60:40.
    if there is no consideration of cost, you just do it as increase equity weight by 10%.
    However considering cost, you have to buy equity by 9.972% and sell fixed income by 10.012%.
    this function is for.
    :param w1: ex_weight
    :param w2: post_weight
    :param cost: cost rate
    :return: weight changes
    :rtype: list
    """
    w = [ww2-ww1 for ww1, ww2 in zip(w1, w2)]
    if w[0]*w[1] > 0:
        raise Exception("it is impossible.")
    if w[0] > 0:
        a = 1 + w2[0]*cost
        b = w2[0]*(-cost)
        c = w2[1]*cost
        d = 1 + w2[1]*(-cost)
    else:
        a = 1 + w2[0]*(-cost)
        b = w2[0]*cost
        c = w2[1]*(-cost)
        d = 1 + w2[1]*cost
    size = a*d - b*c
    diff = [(d*w[0] - b*w[1])/size, (-c*w[0] + a*w[1])/size]
    return diff


if __name__ == "__main__":
    print(cost_balancing([0.6, 0.4], [0.7, 0.3], 0.002))
