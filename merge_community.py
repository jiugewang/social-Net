import cnn_socialNet_read_data
import evaluation
import nmi_test
social_list = [['1', '105', '42', '17', '5', '10', '94', '24'], ['28', '97', '66', '71', '18', '57', '63', '77', '96', '21', '114', '88'], ['98', '64'], ['61', '3', '107', '48', '65', '14', '40', '7', '16', '101', '33'], ['108', '11', '6', '99', '103', '85', '73', '41', '82', '75', '4', '53'], ['90', '38', '110', '26', '106', '46', '34', '104', '2'], ['84', '74', '111', '50', '54', '68', '47', '115', '89'], ['113', '92', '67', '76', '58', '45', '93', '49', '87'], ['59', '60'], ['112', '78', '22', '52', '69', '9', '8', '79', '109', '23'], ['102', '95', '36', '31', '30', '20', '56', '80', '81', '83'], ['51', '12', '25', '70', '91', '29'], ['55', '39', '32', '72', '15', '100', '19', '62', '35', '27', '44', '13', '86', '43', '37']]


def merge_community(comm_list, G):
    """
    合并社区
    :param comm_list: [[1,2,3],[4,5,6],...]
    :param G: 原始图nx.Graph()
    :return: 合并后的社区列表[[1,2,3],[4,5,6],...]
    """
    real_social = []
    virtual_social = []
    for comm in comm_list:
        if len(comm) > 5:
            real_social.append(comm)
        else:
            virtual_social.append(comm)
    matrix_r = [[0 for i in range(len(real_social))] for i in range(len(virtual_social))]

    for i in range(len(virtual_social)):
        for j in range(len(real_social)):
            merge_com = virtual_social[i]+real_social[j]
            virtual_comm = virtual_social[i]
            virtual_r = evaluation.R(virtual_comm, G)
            merge_r = evaluation.R(merge_com, G)
            data_r = merge_r - virtual_r
            matrix_r[i][j] = data_r
    print(matrix_r)
    print(len(matrix_r))
    for m in range(len(matrix_r)):
        row_list = matrix_r[m]
        row = row_list.index(max(row_list))
        print(row_list)
        print(row)
        real_social[row] + virtual_social[m]
        #real_social[matrix_r[m].index(max(matrix_r[m]))]+(virtual_social[m])
        #print(matrix_r[m])
    return real_social


football_network = './small_data/football-edges.txt'
G = cnn_socialNet_read_data.get_graph(football_network,split=',')
print(merge_community(social_list, G))
#print(len(merge_community(social_list, G)))
#print('nmi:',nmi_test.nmi())