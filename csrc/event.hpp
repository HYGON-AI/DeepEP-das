// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include <ATen/hip/HIPContext.h>
#include "kernels/exception.cuh"

namespace deep_ep {

using HIPStream = decltype(at::hip::getCurrentHIPStreamMasqueradingAsCUDA());

struct EventHandle {
    std::shared_ptr<torch::Event> event;

    EventHandle() {
        event = std::make_shared<torch::Event>(torch::kCUDA);
        event->record(at::hip::getCurrentHIPStreamMasqueradingAsCUDA());
    }

    explicit EventHandle(const HIPStream &stream) {
        event = std::make_shared<torch::Event>(torch::kCUDA);
        event->record(stream);
    }

    EventHandle(const EventHandle &other) = default;

    void current_stream_wait() const {
        at::hip::getCurrentHIPStreamMasqueradingAsCUDA().unwrap().wait(*event);
    }
};

inline torch::Event create_event(const HIPStream &s) {
    auto event = torch::Event(torch::kCUDA);
    event.record(s);
    return event;
}

inline void stream_wait(const HIPStream &s_0, const HIPStream &s_1) {
    EP_HOST_ASSERT(s_0.id() != s_1.id());
    s_0.unwrap().wait(create_event(s_1));
}

inline void stream_wait(const HIPStream &s, const EventHandle &event) {
    s.unwrap().wait(*event.event);
}

} // namespace deep_ep
