import torch
import torch.nn as nn
import torch.nn.functional as F

class resnet18(nn.Module):

    def __init__(self, num_classes=10):
        super(resnet18, self).__init__()
        # layer input
        self.conv_input = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(64)
        
        # layer stage 1
        self.conv_1_1_stage_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1_1_stage_1 = nn.BatchNorm2d(64)

        self.conv_2_1_stage_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_2_1_stage_1 = nn.BatchNorm2d(64)
        self.conv_1_2_stage_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1_2_stage_1 = nn.BatchNorm2d(64)
        self.conv_2_2_stage_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_2_2_stage_1 = nn.BatchNorm2d(64)
        # layer stage 2
        self.conv_1_1_stage_2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_1_1_stage_2 = nn.BatchNorm2d(128)

        self.conv_2_1_stage_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1_2_stage_2 = nn.BatchNorm2d(128)
        self.conv_1_2_stage_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_2_1_stage_2 = nn.BatchNorm2d(128)
        self.conv_2_2_stage_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_2_2_stage_2 = nn.BatchNorm2d(128)
        self.res_stage_2_1 = nn.Sequential(
                    nn.Conv2d(64, 128,kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(128)
                    )
        self.res_stage_2_2 = nn.Sequential(
                    nn.Conv2d(128, 128,kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(128)
                    )

        # layer stage 3
        self.conv_1_1_stage_3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_1_1_stage_3 = nn.BatchNorm2d(256)

        self.conv_2_1_stage_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_2_1_stage_3 = nn.BatchNorm2d(256)
        self.conv_1_2_stage_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1_2_stage_3 = nn.BatchNorm2d(256)
        self.conv_2_2_stage_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_2_2_stage_3 = nn.BatchNorm2d(256)
        self.res_stage_3_1 = nn.Sequential(
                    nn.Conv2d(128, 256,kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(256)
                    )
        self.res_stage_3_2 = nn.Sequential(
                    nn.Conv2d(256, 256,kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(256)
                    )

        # layer stage 4
        self.conv_1_1_stage_4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_1_1_stage_4 = nn.BatchNorm2d(512)
        
        self.conv_2_1_stage_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_2_1_stage_4 = nn.BatchNorm2d(512)
        self.conv_1_2_stage_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1_2_stage_4 = nn.BatchNorm2d(512)
        self.conv_2_2_stage_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_2_2_stage_4 = nn.BatchNorm2d(512)
        self.res_stage_4_1 = nn.Sequential(
                    nn.Conv2d(256, 512,kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(512)
                    )
        self.res_stage_4_2 = nn.Sequential(
                nn.Conv2d(512, 512,kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(512)
                )

        # layer output
        self.linear = nn.Linear(512, num_classes)
    
    def forward(self, input_value, old_res_list, is_time_res):
        
        new_res_list = []
        idx = 0
        # inpute stage
        output = F.relu(self.bn_input(self.conv_input(input_value)))

        # layer stage 1
        res = output
        output = F.relu(self.bn_1_1_stage_1(self.conv_1_1_stage_1(output)))
        output = self.bn_2_1_stage_1(self.conv_2_1_stage_1(output))
        output += res

        if is_time_res:
            output += old_res_list[idx]
            idx += 1
        new_res_list.append(output)
        output = F.relu(output)

        res = output
        output = F.relu(self.bn_1_2_stage_1(self.conv_1_2_stage_1(output)))
        output = self.bn_2_2_stage_1(self.conv_2_2_stage_1(output))
        output += res
        if is_time_res:
            output += old_res_list[idx]
            idx += 1
        new_res_list.append(output)
        output = F.relu(output)

        # layer stage 2
        res = self.res_stage_2_1(output)
        output = F.relu(self.bn_1_1_stage_2(self.conv_1_1_stage_2(output)))
        output = self.bn_2_1_stage_2(self.conv_2_1_stage_2(output))
        output += res
        if is_time_res:
            output += old_res_list[idx]
            idx += 1
        new_res_list.append(output)
        output = F.relu(output)

        res = self.res_stage_2_2(output)
        output = F.relu(self.bn_1_2_stage_2(self.conv_1_2_stage_2(output)))
        output = self.bn_2_2_stage_2(self.conv_2_2_stage_2(output))
        output += res
        if is_time_res:
            output += old_res_list[idx]
            idx += 1
        new_res_list.append(output)
        output = F.relu(output)

        # layer stage 3
        res = self.res_stage_3_1(output)
        output = F.relu(self.bn_1_1_stage_3(self.conv_1_1_stage_3(output)))
        output = self.bn_2_1_stage_3(self.conv_2_1_stage_3(output))
        output += res
        if is_time_res:
            output += old_res_list[idx]
            idx += 1
        new_res_list.append(output)
        output = F.relu(output)

        res = self.res_stage_3_2(output)
        # new_res_list.append(res)
        output = F.relu(self.bn_1_2_stage_3(self.conv_1_2_stage_3(output)))
        output = self.bn_2_2_stage_3(self.conv_2_2_stage_3(output))
        output += res
        if is_time_res:
            output += old_res_list[idx]
            idx += 1
        new_res_list.append(output)
        output = F.relu(output)

        # layer stage 4
        res = self.res_stage_4_1(output)
        output = F.relu(self.bn_1_1_stage_4(self.conv_1_1_stage_4(output)))
        output = self.bn_2_1_stage_4(self.conv_2_1_stage_4(output))
        output += res
        if is_time_res:
            output += old_res_list[idx]
            idx += 1
        new_res_list.append(output)
        output = F.relu(output)

        res = self.res_stage_4_2(output)
        output = F.relu(self.bn_1_2_stage_4(self.conv_2_1_stage_4(output)))
        output = self.bn_2_2_stage_4(self.conv_2_2_stage_4(output))
        output += res
        if is_time_res:
            output += old_res_list[idx]
            idx += 1
        new_res_list.append(output)
        output = F.relu(output)

        # output stage
        output = F.avg_pool2d(output, 4)
        output = output.view(output.size(0), -1)
        output = self.linear(output)

        # for i in new_res_list:
        #     print(i.shape)
        return output, new_res_list