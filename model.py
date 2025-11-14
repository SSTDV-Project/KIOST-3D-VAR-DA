import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):

        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            hidden_state = [hidden_state]
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class DecBlock(nn.Module):
    def __init__(self, channels):
        super(DecBlock, self).__init__()

        self.bn1   = nn.BatchNorm2d(channels[0], affine=False)
        self.bnA   = nn.BatchNorm2d(channels[0])
        self.fc1   = nn.Linear(48, 2 * channels[0])
        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels[1], affine=False)
        self.bnB   = nn.BatchNorm2d(channels[1])
        self.fc2   = nn.Linear(48, 2 * channels[1])
        self.conv2 = nn.ConvTranspose2d(channels[1], channels[2], kernel_size=4, padding=1, stride=2)
        self.conv3 = nn.ConvTranspose2d(channels[0], channels[2], kernel_size=2, stride=2)

    def forward(self, x, t=None):
        if t is None:
            y = self.conv3(x)

            x = self.conv1(F.relu(self.bnA(x), inplace=True))
            x = self.conv2(F.relu(self.bnB(x), inplace=True))
        else:
            y = self.conv3(x)
        
            gamma, beta = self.fc1(t).chunk(2, 1)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta  = beta.unsqueeze(-1).unsqueeze(-1)
            x = self.conv1(F.relu(gamma * self.bn1(x) + beta, inplace=True))
            gamma, beta = self.fc2(t).chunk(2, 1)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta  = beta.unsqueeze(-1).unsqueeze(-1)
            x = self.conv2(F.relu(gamma * self.bn2(x) + beta, inplace=True))
        
        return x + y

class EncBlock(nn.Module):
    def __init__(self, channels):
        super(EncBlock, self).__init__()

        self.bnA   = nn.BatchNorm2d(channels[0])
        self.bn1   = nn.BatchNorm2d(channels[0], affine=False)
        self.fc1   = nn.Linear(48, channels[0] * 2)
        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=4, padding=1, stride=2)
        self.bnB   = nn.BatchNorm2d(channels[1])
        self.bn2   = nn.BatchNorm2d(channels[1], affine=False)
        self.fc2   = nn.Linear(48, channels[1] * 2)
        self.conv2 = nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels[0], channels[1], kernel_size=2, stride=2)

    def forward(self, x, t=None):
        if t is None:
            y = self.conv3(x)

            x = self.conv1(F.relu(self.bnA(x), inplace=True))
            x = self.conv2(F.relu(self.bnB(x), inplace=True))
        else:
            y = self.conv3(x)

            gamma, beta = self.fc1(t).chunk(2, 1)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta  = beta.unsqueeze(-1).unsqueeze(-1)
            x = self.conv1(F.relu(gamma * self.bn1(x) + beta, inplace=True))
            gamma, beta = self.fc2(t).chunk(2, 1)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta  = beta.unsqueeze(-1).unsqueeze(-1)
            x = self.conv2(F.relu(gamma * self.bn2(x) + beta, inplace=True))

        return x + y

class Decoder(nn.Module):
    def __init__(self, block):
        super(Decoder, self).__init__()

        self.lstm = ConvLSTM(input_dim=2+32+32, hidden_dim=32, kernel_size=(1, 1), num_layers=2)
        self.conv = nn.Conv2d(64, 32, kernel_size=1)

        self.forward1 = nn.ModuleList()
        self.forward1.append(nn.Conv2d(32, 32, kernel_size=1))
        self.forward1.append(EncBlock([ 32,  64,  64]))
        self.forward1.append(EncBlock([ 64, 128, 128]))
        self.forward1.append(EncBlock([128, 256, 256]))
        self.forward1.append(EncBlock([256, 512, 512]))

        self.forward2 = nn.ModuleList()
        self.forward2.append(DecBlock([512, 512, 256]))
        self.forward2.append(DecBlock([512, 256, 128]))
        self.forward2.append(DecBlock([256, 128,  64]))
        self.forward2.append(DecBlock([128,  64,  32]))

        self.forwardA = nn.Sequential(
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 2, kernel_size=3, padding=1))
        self.forwardB = nn.Sequential(
                block([32, 32, 16]),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 2, kernel_size=3, padding=1))
    
    def forward(self, x, t, batch_size):
        x = x.view(batch_size, x.size(0) // batch_size, x.size(1), x.size(2), x.size(3))  # batch_size, time_step, channel, height, width
        x = x.permute(1, 0, 2, 3, 4)                                                      # time_step, batch_size, channel, height, width
        x = self.lstm(x)[0][-1][:, -1]

        x1 = self.forward1[0](x)
        x2 = self.forward1[1](x1, t)
        x3 = self.forward1[2](x2, t)
        x4 = self.forward1[3](x3, t)
        x5 = self.forward1[4](x4, t)
        
        x5 = self.forward2[0](x5, t)
        x4 = self.forward2[1](torch.cat([x4, x5], dim=1), t)
        x3 = self.forward2[2](torch.cat([x3, x4], dim=1), t)
        x2 = self.forward2[3](torch.cat([x2, x3], dim=1), t)
        x1 = F.leaky_relu(torch.cat([x1, x2], dim=1))
        x  = self.conv(x1)

        wave = self.forwardA(x)[:, :, :34, :30]
        wind = self.forwardB(x)[:, :, :68, :60]

        return wave[:, 0:1], wave[:, 1:2], wind[:, 0:1], wind[:, 1:2]

class Encoder(nn.Module):
    def __init__(self, block):
        super(Encoder, self).__init__()

        self.forward1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                block([16, 32, 32]))
        self.forward2 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                block([16, 32, 32]))

    def forward(self, swh, mwp, u_wind, v_wind):
        swh[torch.isnan(swh)] = 0
        swh = swh.contiguous().view(swh.size(0) * swh.size(1), swh.size(2), swh.size(3), swh.size(4))
        swh = F.pad(swh, (0, 18, 0, 14))
        
        mwp[torch.isnan(mwp)] = 0
        mwp = mwp.contiguous().view(mwp.size(0) * mwp.size(1), mwp.size(2), mwp.size(3), mwp.size(4))
        mwp = F.pad(mwp, (0, 18, 0, 14))
        
        u_wind[torch.isnan(u_wind)] = 0
        u_wind = u_wind.contiguous().view(u_wind.size(0) * u_wind.size(1), u_wind.size(2), u_wind.size(3), u_wind.size(4))
        u_wind = F.pad(u_wind, (0, 36, 0, 28))
        
        v_wind[torch.isnan(v_wind)] = 0
        v_wind = v_wind.contiguous().view(v_wind.size(0) * v_wind.size(1), v_wind.size(2), v_wind.size(3), v_wind.size(4))
        v_wind = F.pad(v_wind, (0, 36, 0, 28))

        # time = time.view(time.size(0), 1, time.size(1), 1, 1).expand(time.size(0), wave.size(1), time.size(1), wave.size(3), wave.size(4))
        # wave = torch.cat([wave, time], dim=2)

        x = torch.cat([self.forward1(u_wind), self.forward2(v_wind)], dim=1)
        x = torch.cat([swh, mwp, x], dim=1)

        return x
