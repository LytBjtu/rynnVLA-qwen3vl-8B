import logging
import socket


def patch_torch_elastic_for_deepspeed():
    """
    DeepSpeed still imports deprecated symbols from
    `torch.distributed.elastic.agent.server.api` on some torch versions.
    Patch them back before importing deepspeed.
    """
    try:
        from torch.distributed.elastic.agent.server import api as elastic_api
    except Exception:
        return

    if not hasattr(elastic_api, "log"):
        elastic_api.log = getattr(elastic_api, "logger", logging.getLogger("torch.distributed.elastic"))

    if not hasattr(elastic_api, "_get_socket_with_port"):
        def _get_socket_with_port():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("", 0))
            sock.listen(0)
            return sock

        elastic_api._get_socket_with_port = _get_socket_with_port
