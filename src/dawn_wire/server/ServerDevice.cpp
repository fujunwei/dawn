// Copyright 2019 The Dawn Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "dawn_wire/server/Server.h"

namespace dawn_wire { namespace server {

    void Server::ForwardUncapturedError(DawnErrorType type, const char* message, void* userdata) {
        auto server = static_cast<Server*>(userdata);
        server->OnUncapturedError(type, message);
    }

    void Server::OnUncapturedError(DawnErrorType type, const char* message) {
        ReturnDeviceUncapturedErrorCallbackCmd cmd;
        cmd.type = type;
        cmd.message = message;

        size_t requiredSize = cmd.GetRequiredSize();
        char* allocatedBuffer = static_cast<char*>(GetCmdSpace(requiredSize));
        cmd.Serialize(allocatedBuffer);
    }

    bool Server::DoDevicePopErrorScope(DawnDevice cDevice, uint64_t requestSerial) {
        ErrorScopeUserdata* userdata = new ErrorScopeUserdata;
        userdata->server = this;
        userdata->requestSerial = requestSerial;

        return mProcs.devicePopErrorScope(cDevice, ForwardPopErrorScope, userdata);
    }

    // static
    void Server::ForwardPopErrorScope(DawnErrorType type, const char* message, void* userdata) {
        auto* data = reinterpret_cast<ErrorScopeUserdata*>(userdata);
        data->server->OnDevicePopErrorScope(type, message, data);
    }

    void Server::OnDevicePopErrorScope(DawnErrorType type,
                                       const char* message,
                                       ErrorScopeUserdata* userdata) {
        std::unique_ptr<ErrorScopeUserdata> data{userdata};

        ReturnDevicePopErrorScopeCallbackCmd cmd;
        cmd.requestSerial = data->requestSerial;
        cmd.type = type;
        cmd.message = message;

        size_t requiredSize = cmd.GetRequiredSize();
        char* allocatedBuffer = static_cast<char*>(GetCmdSpace(requiredSize));
        cmd.Serialize(allocatedBuffer);
    }

}}  // namespace dawn_wire::server
