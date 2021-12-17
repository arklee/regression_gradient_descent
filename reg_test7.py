from regression import problem

# 目标函数，th为特征值集合，x为自变量集合
def h(th, x):
    return th[0] * x[0] + th[1] * x[1] + th[2] * x[2] + th[3]

# 输入矩阵，input[i][j]表示第i+1个训练集的第j+1个自变量
input = [[0.6891284936700826, -1.781583166457379, -2.512561916725675], [0.845398588572015, 0.22544974146614383, 3.450043133984706], [3.5350813395332104, 1.3175336439669305, 2.1931388204537727], [-2.0734682733852927, 3.615867043046431, -0.17048534398298454], [-4.659143694121816, -0.13366479941379517, -2.9329086187761213], [3.5289615542499853, -1.9561900261099663, 0.587249526889605], [0.9730888984521491, -2.7610731190773183, 1.8690027354527294], [-3.6199214747049844, -0.2688257656579762, -2.7463678965337657], [0.8808742274115411, 2.713335746082678, 1.83311800985968], [3.7342007327272073, -0.20047888099771982, -0.7177419014154329], [-3.591231255473782, 3.021194155209528, 2.9630030739002198], [-1.4875884826073127, 2.2969400428205446, -2.453889415106152], [-0.4445324319310273, -3.1399291535477794, 1.6467396361131736], [0.336791111080571, 0.7371887053840864, -1.0098258492697487], [3.934149672317159, -1.332498417895045, -0.9919424533715571], [-4.882501468275389, -1.4877600475512875, 2.005076377327797], [4.150117462721436, 1.259904774817936, 1.4388910928072218], [4.419206718773351, 1.5687151514437003, -2.5252571425185244], [-0.4335527934586947, -4.702690439237813, -4.433356454081615], [3.758433185933633, -2.3412373498623937, -0.9773084861086623], [2.225458719883372, -1.5691327894251461, 0.6120468763796949], [-4.21864122299787, 0.17973152493971578, 1.5852935786994593], [-0.8435059374061937, -0.6219296035973287, -2.105784002286998], [-1.6872453175211044, 0.2655526321882651, 1.167840513548405], [4.951105925840195, -2.822869775025091, -2.5678362830459243], [3.4072631450309796, -1.67642760265037, 3.6594786465832643], [-4.810315093164617, 2.2640105818708376, -1.6616010564605277], [-3.001685674890987, 0.36989511762958505, -3.138192323799589], [2.5957308034070814, -4.311554017453468, -2.3009665625538602], [1.6681873511778922, 3.5900764076123917, -2.9364562297760077], [3.995965321980485, 3.3982025499327886, 4.912090202379251], [2.6689886508431093, -0.37787748167290025, 4.6795984971852755], [4.7169105977498464, -2.7358959606658195, 3.0911762172051405], [0.1881139202353077, 0.6694313492232717, 2.542229359144539], [3.342981452740262, 2.9690715012795144, -1.6750747484016126], [-3.5951735279276242, -0.06610276578222551, 1.5674109619678456], [3.6585369821720857, 1.2748614839923467, -3.6482961833342173], [-0.26275851053539245, 2.71311072539671, 4.104380450652431], [-4.029531298005331, -3.55821245127671, -3.2232560563810377], [4.643165578313143, -2.3931535012449032, 2.145855895418941], [-4.232170717585083, -2.5950678960049665, 3.397080662633778], [1.376364439015958, 0.9792513419695781, 2.9451241531373284], [-1.2668820656736601, 2.939170966235425, 4.578913355795692], [3.0117551255161743, 1.7653286615931385, 4.495333790616842], [-2.8498118540857433, -3.387828790205869, 3.4910162411769563], [-0.6858033162869082, -0.08430423231328033, 1.493553452506241], [2.3140647718926397, 4.420334374115956, -3.4650046116677413], [4.852286166946202, 1.4648855766362967, 1.1408432046856765], [-2.9138753168026543, -4.351628705631265, 3.5817826549137677], [3.646378351917443, 0.6835514579848112, -1.366362015626107]]

# 输出向量
output = [0.022932959028500788, -4.009126562299475, 14.541963349101712, 13.358105139341916, -7.408365167033338, -0.4752052202834426, -16.389368242621178, -7.986698729302159, 10.311574250647785, 12.789592842150945, 0.8369539028640651, 9.605081770712673, -19.232729052284167, 5.643390212840485, 4.7289230939698665, -27.53994357964379, 17.60714936897834, 24.992204175070697, -15.416827765583701, 1.3405777251441746, -3.3820313715825012, -15.38654834092704, -3.131608409431915, -3.98552042707364, 6.9821202576528165, -7.2117191363345, -1.337093903626609, -0.6528589504585263, -9.436086810008259, 28.89644044601295, 19.561263292819874, -3.5723485716073284, -7.372705968214062, -0.8129301395265864, 27.446367037840343, -16.703730698924584, 25.643160594502948, 5.467702869219069, -21.830644091552276, -4.71942566441255, -30.859327317879128, 2.750600006833804, 1.063465372910585, 10.855641051861046, -33.33822196592601, -3.444052989755872, 37.08507104552488, 17.72596773244225, -36.47340010022797, 15.596986895547861]

# problem(输入矩阵, 输出向量, 目标函数, 迭代次数, 特征值个数)
p = problem(input, output, h, 1000, 4)

p.solve()
p.show()