# revolve-recovery: the datagen image as is. Recovery plans the rest of a task from a saved phase state
# with the same AutoMoMa + cuRobo plan step (plan_from_state.py), so it needs exactly datagen's
# dependencies; nothing is added. cuRobo (ThirdParty/curobo/install) and AutoMoMa (ThirdParty/automoma)
# are mounted at runtime as for datagen (`./revolve build datagen` builds them). WORKDIR /repo and CMD
# are inherited.
ARG DATAGEN_IMAGE=revolve-datagen:latest
FROM ${DATAGEN_IMAGE}
LABEL org.revolve.component=recovery
