"""Combine testing results of the three models to get final accuracy."""

import argparse
import numpy as np

def fusion_sort(score):
    score_sort = np.argsort(np.argsort(score))
    score=score+score_sort
    score=score*0.5

    return score



def main():
    parser = argparse.ArgumentParser(description="combine predictions")
    parser.add_argument('--iframe', type=str, required=True,
                        help='iframe score file.')
    parser.add_argument('--mv', type=str, required=True,
                        help='motion vector score file.')
    parser.add_argument('--res', type=str, required=True,
                        help='residual score file.')
    parser.add_argument('--glocal', type=float, default=0,
                        help='residual weight.')

    parser.add_argument('--wi', type=float, default=1,
                        help='iframe weight.')
    parser.add_argument('--wm', type=float, default=1,
                        help='motion vector weight.')
    parser.add_argument('--wr', type=float, default=4,
                        help='residual weight.')

    args = parser.parse_args()

    with np.load(args.iframe,allow_pickle=True) as iframe:
        with np.load(args.mv,allow_pickle=True) as mv:
            with np.load(args.res,allow_pickle=True) as residual:
                n = len(mv['names'])
                flag=1

                i_score = np.array([score[0][0] for score in iframe['scores']])
                # i_score=fusion_sort(i_score)



                mv_score = np.array([score[0][0] for score in mv['scores']])
                # mv_score=fusion_sort(mv_score)



                res_score = np.array([score[0][0] for score in residual['scores']])
                # res_score=fusion_sort(res_score)

                i_label = np.array([score[1] for score in iframe['scores']])
                mv_label = np.array([score[1] for score in mv['scores']])
                res_label = np.array([score[1] for score in residual['scores']])
                assert np.alltrue(i_label == mv_label) and np.alltrue(i_label == res_label)
                # i_x=i_score * args.wi
                # m_x=mv_score * args.wm
                # r_x=res_score * args.wr
                acc_max=0
                if args.glocal == 0:
                    for t in range(20):
                        T_w=1
                        for m in range(20):
                            mv_w=0
                            for r in range(20):
                                R_w=0
                                # combined_score = i_score * args.wi + mv_score * args.wm + res_score * args.wr
                                combined_score = i_score * T_w+ mv_score * mv_w+ res_score * R_w
                                accuracy = float(sum(np.argmax(combined_score, axis=1) == i_label)) / n
                                if accuracy>acc_max:
                                    acc_max=accuracy
                                    i_max=T_w
                                    mv_max=mv_w
                                    r_max=R_w

                # T_w=1
                # for m in range(20):
                #     mv_w=m
                #     for r in range(20):
                #         R_w=r
                #         # combined_score = i_score * args.wi + mv_score * args.wm + res_score * args.wr
                #         combined_score = i_score * T_w+ mv_score * mv_w+ res_score * R_w
                #         accuracy = float(sum(np.argmax(combined_score, axis=1) == i_label)) / n
                #         if accuracy>=acc_max:
                #             acc_max=accuracy
                #             i_max=T_w
                #             mv_max=mv_w
                #             r_max=R_w
                #             print('Accuracy: %f (%d).' % (acc_max, n))
                #             print('i: %d,mv:%d,r:%d,' % (i_max, mv_max, r_max))

                # if args.glocal == 1:
                #     for t in range(20):
                #         T_w=1
                #         for m in range(20):
                #             mv_w=m
                #             for r in range(20):
                #                 R_w=r
                #                 # combined_score = i_score * args.wi + mv_score * args.wm + res_score * args.wr
                #                 combined_score = i_score * T_w+ mv_score * mv_w+ res_score * R_w
                #                 accuracy = float(sum(np.argmax(combined_score, axis=1) == i_label)) / n
                #                 if accuracy>=acc_max:
                #                     acc_max=accuracy
                #                     i_max=T_w
                #                     mv_max=mv_w
                #                     r_max=R_w
                #                     print('Accuracy: %f (%d).' % (acc_max, n))
                #                     print('i: %d,mv:%d,r:%d,' % (i_max, mv_max, r_max))



                if args.glocal == 2:
                    for t in range(20):
                        T_w=0
                        for m in range(20):
                            mv_w=1
                            for r in range(20):
                                R_w=1
                                # combined_score = i_score * args.wi + mv_score * args.wm + res_score * args.wr
                                combined_score = i_score * T_w+ mv_score * mv_w+ res_score * R_w
                                accuracy = float(sum(np.argmax(combined_score, axis=1) == i_label)) / n
                                if accuracy>acc_max:
                                    acc_max=accuracy
                                    i_max=T_w
                                    mv_max=mv_w
                                    r_max=R_w

                print('Accuracy: %f (%d).' % (acc_max, n))
                print('i: %d,mv:%d,r:%d,' % (i_max,mv_max,r_max))

if __name__ == '__main__':

    main()
