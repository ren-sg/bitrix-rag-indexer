from __future__ import annotations


class OnnxRuntimeEnvironment:
    """Configures ONNX Runtime before FastEmbed creates inference sessions."""

    def __init__(
        self,
        *,
        cuda: bool,
        log_severity: int = 3,
        preload_cuda_dependencies: bool = True,
    ) -> None:
        self.cuda = cuda
        self.log_severity = log_severity
        self.preload_cuda_dependencies = preload_cuda_dependencies

    def apply(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            return

        self._configure_logging(ort)

        if self.cuda and self.preload_cuda_dependencies:
            self._preload_cuda_dependencies(ort)

    def _configure_logging(self, ort: object) -> None:
        set_default_logger_severity = getattr(
            ort,
            "set_default_logger_severity",
            None,
        )
        if set_default_logger_severity is None:
            return

        set_default_logger_severity(int(self.log_severity))

    def _preload_cuda_dependencies(self, ort: object) -> None:
        preload_dlls = getattr(ort, "preload_dlls", None)
        if preload_dlls is None:
            return

        preload_dlls(directory="")
