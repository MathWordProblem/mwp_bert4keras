from train import get_answer


def predict(in_file, out_file, topk=1):
    """输出预测结果到文件
    该函数主要为比赛 https://www.datafountain.cn/competitions/467 所写，
    主要是读取该比赛的测试集，然后预测equation，并且根据不同的问题输出不同格式的答案，
    out_file可以直接提交到线上评测，线上准确率可以达到38%+。
    """
    fw = open(out_file, 'w', encoding='utf-8')
    raw_data = pd.read_csv(in_file, header=None, encoding='utf-8')
    for i, question in tqdm(raw_data.values):
        pred_equation = get_answer(question)
        if '.' not in pred_equation:
            pred_equation = re.sub('([\d]+)', 'Integer(\\1)', pred_equation)
        try:
            pred_answer = eval(pred_equation)
        except:
            pred_answer = np.random.choice(21) + 1
        if '.' in pred_equation:
            if u'百分之几' in question:
                pred_answer = pred_answer * 100
            pred_answer = round(pred_answer, 2)
            if int(pred_answer) == pred_answer:
                pred_answer = int(pred_answer)
            if (
                re.findall(u'多少[辆|人|个|只|箱|包本|束|头|盒|张]', question) or
                re.findall(u'几[辆|人|个|只|箱|包|本|束|头|盒|张]', question)
            ):
                if re.findall(u'至少|最少', question):
                    pred_answer = np.ceil(pred_answer)
                elif re.findall(u'至多|最多', question):
                    pred_answer = np.floor(pred_answer)
                else:
                    pred_answer = np.ceil(pred_answer)
                pred_answer = int(pred_answer)
            pred_answer = str(pred_answer)
            if u'百分之几' in question:
                pred_answer = pred_answer + '%'
        else:
            pred_answer = str(pred_answer)
            if '/' in pred_answer:
                if re.findall('\d+/\d+', question):
                    a, b = pred_answer.split('/')
                    a, b = int(a), int(b)
                    if a > b:
                        pred_answer = '%s_%s/%s' % (a // b, a % b, b)
                else:
                    if re.findall(u'至少|最少', question):
                        pred_answer = np.ceil(eval(pred_answer))
                    elif re.findall(u'至多|最多', question):
                        pred_answer = np.floor(eval(pred_answer))
                    else:
                        pred_answer = np.ceil(eval(pred_answer))
                    pred_answer = str(int(pred_answer))
        fw.write(str(i) + ',' + pred_answer + '\n')
        fw.flush()
    fw.close()



if name == '__main__':
    predict('dataset/contest/test.csv', 'dataset/contest/result.csv')

