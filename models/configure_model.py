from models.decoder import SingleGaussianDecoderStd, SingleGaussianDecoderStdParameter, MixtureDensityDecoder
from models.encoders import EncoderRNN, ContrastiveEncoderRNN, ModifiedContrastiveEncoderRNN
from models.model import Model
from models.multi_head_mog import *


# from models.gnn.gnn_post_lstm import GNNPostLSTM

def configure_decoder(conf, dim, num_heads, multi_head):
    number_gaussians = conf["number_gaussians"]
    # if multi_head:
    if multi_head:
        if conf["model_type"] == "vrnn" or conf["model_type"] == "variational_gnn":
            from models.variational_rnn_decoder import VariationalMixtureDecoderMultiHead
            decoder = VariationalMixtureDecoderMultiHead(
                input_dim=dim,
                num_heads=num_heads,
                output_dim=2,
                num_gaussians=number_gaussians,
                kl_loss_wt=conf["kl_loss_weight"]
            )
            return decoder

    if conf["model_type"] == "contrastive_vector":
        loss_type = conf["contrastive_loss"]
        decoder = ModifiedContrastiveMixureDecoderMultiHead(
            input_dim=dim,
            num_heads=num_heads,
            output_dim=2,
            num_gaussians=number_gaussians,
            loss_type=loss_type
        )

    elif conf["model_type"] == "contrastive_gnn" or conf["model_type"] == "simclr_gnn" or conf[
        "model_type"] == "cpc_gnn":
        loss_type = conf["contrastive_loss"]
        if not multi_head and num_heads > 1:
            decoder = Seq2SeqContrastiveAutoregressiveDecoder(input_dim=dim,
                                                              num_heads=num_heads,
                                                              output_dim=2,
                                                              num_gaussians=number_gaussians)

        else:
            decoder = ContrastiveGNNMixureDecoderMultiHead(
                input_dim=dim,
                num_heads=num_heads,
                output_dim=2,
                num_gaussians=number_gaussians,
                loss_type=loss_type
            )

        # else:
        #     decoder = MixureDecoderMultiHead(
        #         input_dim=dim,
        #         num_heads=num_heads,
        #         output_dim=2,
        #         num_gaussians=number_gaussians,
        #     )
    else:
        decoder = MixureDecoderMultiHead(
            input_dim=dim,
            num_heads=num_heads,
            output_dim=2,  # output dimension is always 2 since this is in the middle
            num_gaussians=number_gaussians,
        )

    return decoder


def configure_mlp(conf, num_heads, multi_head):
    input_dim = conf["input_dim"]
    # from models.decoder import MixtureDensityMLP
    # decoder = MixtureDensityMLP(input_dim=input_dim, hidden=conf["hidden_dim"],
    #                           output_dim=2, num_gaussians=conf["number_gaussians"],
    #                           non_linear=nn.ReLU())
    # encoder = None
    from models.mlp import MixtureDensityMLP
    model = MixtureDensityMLP()
    return model

def configure_vgnn_model(conf, num_heads, total_input_dim, multi_head, concat_dim):
    """
    Concat dimension is what we concatenate to the final gnn pooled state that we feed into decoder
    Use variational RNN instead of regular LSTM
    """
    from models.gnn.variational_gnn_encoder import VariationalGNN
    # hidden_dim = conf["hidden_dim"]
    # input_dim = 3
    # hidden_dim = 8
    hidden_dim = conf["hidden_dim"]
    gnn_hidden_dim = conf["gnn_hidden_dim"]
    hideout_timestep_dim = 3
    encoder = VariationalGNN(total_input_dim, hidden_dim, gnn_hidden_dim,
                             use_last_k_detections=conf["use_last_k_detections"])
    # dim = gnn_hidden_dim + hideout_timestep_dim

    dim = gnn_hidden_dim + concat_dim

    if conf["use_last_k_detections"]:
        dim += 24  # k * 3 (t, x, y) for the detections

    decoder = configure_decoder(conf, dim, num_heads, multi_head)
    model = Model(encoder, decoder)
    return model

def configure_vrnn(conf, num_heads, multi_head):
    from models.variational_rnn_encoder import VariationalRNNEncoder
    from models.variational_rnn_decoder import VariationalMixtureDecoderMultiHead
    hidden_dim = conf["hidden_dim"]
    number_gaussians = conf["number_gaussians"]
    if conf["model_type"] == "vrnn_padded":
        padded = True
    else:
        padded = False

    encoder = VariationalRNNEncoder(
        input_dim=conf["input_dim"],
        hidden_dim=hidden_dim,
        z_dim=conf["z_dim"],
        num_layers=1,
        padded_input=padded)

    decoder = VariationalMixtureDecoderMultiHead(
                input_dim=hidden_dim + hidden_dim,
                num_heads=num_heads,
                output_dim=2,
                num_gaussians=number_gaussians,
                kl_loss_wt=conf["kl_loss_weight"],
                padded_input=padded
            )

    model = Model(encoder, decoder)
    return model


def configure_cat_vae(conf, num_heads, multi_head):
    from models.categorical_vae import CategoricalVAE
    from models.multi_head_mog import CategoricalVAEDecoder
    encoder_type = conf["encoder_type"]
    hidden_dim = conf["hidden_dim"]
    latent_dim = conf["latent_dim"]
    categorical_dim = conf["categorical_dim"]
    decoder_type = conf["decoder_type"]
    beta = conf["beta"]
    number_gaussians = conf["number_gaussians"]
    input_dim = conf["input_dim"]

    encoder = CategoricalVAE(input_dim, hidden_dim, latent_dim=latent_dim)

    output_dim = 2

    decoder = CategoricalVAEDecoder(
        input_dim=hidden_dim + latent_dim*categorical_dim,
        num_heads=num_heads,
        output_dim=output_dim,  # output dimension is always 2 since this is in the middle
        num_gaussians=number_gaussians,
        beta=beta,
        categorical_dim=categorical_dim,
        latent_dim = latent_dim
    )

    print(decoder_type)

    model = Model(encoder, decoder)

    return model


def configure_vae(conf, num_heads, multi_head):
    from models.vae_opponent_modeling import VAE_Opponent
    from models.multi_head_mog import VAEOpponentDecoderMultiHead
    encoder_type = conf["encoder_type"]
    hidden_dim = conf["hidden_dim"]
    latent_dim = conf["latent_dim"]
    decoder_type = conf["decoder_type"]
    beta = conf["beta"]
    number_gaussians = conf["number_gaussians"]
    input_dim = conf["input_dim"]

    encoder = VAE_Opponent(input_dim, hidden_dim, latent_dim=latent_dim)

    output_dim = 2

    decoder = VAEOpponentDecoderMultiHead(
            input_dim=hidden_dim+latent_dim,
            num_heads=num_heads,
            output_dim=output_dim,  # output dimension is always 2 since this is in the middle
            num_gaussians=number_gaussians,
            beta=beta
        )

    print(decoder_type)

    model = Model(encoder, decoder)

    return model


def configure_regular(conf, num_heads, multi_head):
    """_summary_

    Args:
        conf (dict): The model configuration dictionary from the yaml config file. 
        num_heads (int): Represents the number of heads of the model.

    Returns:
        _type_: pytorch model
    """
    encoder_type = conf["encoder_type"]
    hidden_dim = conf["hidden_dim"]
    decoder_type = conf["decoder_type"]
    number_gaussians = conf["number_gaussians"]
    input_dim = conf["input_dim"]

    if encoder_type == "lstm":
        encoder = EncoderRNN(input_dim, hidden_dim)

    decoder = configure_decoder(conf, hidden_dim, num_heads, multi_head)
    print(decoder_type)

    model = Model(encoder, decoder)

    return model

def get_input_dim(config):
    total_input_dim = 3
    if config["datasets"]["one_hot_agents"]:
        total_input_dim += 3
    if config["datasets"]["detected_location"]:
        total_input_dim += 2
    if config["datasets"]["timestep"]:
        total_input_dim += 1
    return total_input_dim

def configure_mi_red(conf, output_dim):
    from models.mi.mi_red import RedMI
    input_dim = conf["input_dim"]
    h1 = conf["h1"]
    h2 = conf["h2"]
    embedding_dim = conf["embedding_dim"]
    return RedMI(input_dim, output_dim, embedding_dim, h1=h1, h2=h2)

def configure_mi_blue(conf, output_dim):
    from models.mi.mi_blue import BlueMI, BlueMIGaussian
    input_dim = conf["input_dim"]
    h1 = conf["h1"]
    h2 = conf["h2"]
    embedding_dim = conf["embedding_dim"]
    # return BlueMI(input_dim, output_dim, embedding_dim, h1=h1, h2=h2)
    return BlueMIGaussian(input_dim, output_dim, embedding_dim, h1=h1, h2=h2)

def configure_mog_v2(conf):
    from models.mi.mi_blue import BlueMIMixture
    input_dim = conf["input_dim"]
    h1 = conf["h1"]
    h2 = conf["h2"]
    output_dim = 2
    number_gaussians = conf["number_gaussians"]
    return BlueMIMixture(input_dim, output_dim, num_mixtures=number_gaussians, h1=h1, h2=h2)

def configure_mog_agent(conf):
    from models.mi.mi_blue_agent import BlueMIMixtureGNN
    input_dim = conf["input_dim"]
    h1 = conf["h1"]
    h2 = conf["h2"]
    gnn_hidden_dim = conf["gnn_hidden_dim"]
    output_dim = 2
    number_gaussians = conf["number_gaussians"]
    return BlueMIMixtureGNN(input_dim, output_dim, num_mixtures=number_gaussians, h1=h1, h2=h2, gnn_hidden = gnn_hidden_dim)

def configure_model(config):
    conf = config["model"]
    if conf["model_type"] == "mi_red":
        output_dim = conf["output_dim"]
        return configure_mi_red(conf, output_dim)
    elif conf["model_type"] == "mi_blue":
        output_dim = conf["output_dim"]
        return configure_mi_blue(conf, output_dim)
    elif conf["model_type"] == "padded_mog_v2":
        return configure_mog_v2(conf)
    elif conf["model_type"] == "padded_mog_v2_agent_gnn":
        return configure_mog_agent(conf)


    num_heads = config["datasets"]["num_heads"]
    if config["datasets"]["include_current"]:
        num_heads += 1

    multi_head = config["datasets"]["multi_head"]

    if conf["model_type"] == "variational_gnn":
        total_input_dim = get_input_dim(config)
        if config["datasets"]["get_start_location"]:
            concat_dim = 5
        else:
            concat_dim = 3
        return configure_vgnn_model(conf, num_heads, total_input_dim, multi_head, concat_dim)
    elif conf["model_type"] == "vrnn" or conf["model_type"] == "vrnn_padded":
        return configure_vrnn(conf, num_heads, multi_head)
    elif conf["model_type"] == "mlp":
        return configure_mlp(conf, num_heads, multi_head)
    elif conf["model_type"] == "vae":
        return configure_vae(conf, num_heads, multi_head)
    elif conf["model_type"] == "cat_vae":
        return configure_cat_vae(conf, num_heads, multi_head)
    else:
        return configure_regular(conf, num_heads, multi_head)


def configure_and_load_model(path, device):
    import yaml
    config_path = os.path.join(path, "config.yaml")
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    # device = config["device"]

    model = configure_model(config)
    model.to(device)
    return model, config