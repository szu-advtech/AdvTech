/*
 * Copyright 2018 The Polycube Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Modify these methods with your own implementation

#include "Lbdsr.h"
#include "Lbdsr_dp.h"

Lbdsr::Lbdsr(const std::string name, const LbdsrJsonObject &conf)
    : Cube(conf.getBase(), {getCode(true)}, {}) {
  logger()->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [Lbdsr] [%n] [%l] %v");
  logger()->info("Creating Lbdsr instance");

  if (conf.algorithmIsSet()) {
    setAlgorithm(conf.getAlgorithm());
  }

  addPortsList(conf.getPorts());
  addFrontend(conf.getFrontend());
  addBackend(conf.getBackend());
}

Lbdsr::~Lbdsr() {}

void Lbdsr::update(const LbdsrJsonObject &conf) {
  // This method updates all the object/parameter in Lbdsr object specified in
  // the conf JsonObject.
  // You can modify this implementation.
  Cube::set_conf(conf.getBase());

  if (conf.frontendIsSet()) {
    auto m = getFrontend();
    m->update(conf.getFrontend());
  }

  if (conf.algorithmIsSet()) {
    setAlgorithm(conf.getAlgorithm());
  }

  if (conf.portsIsSet()) {
    for (auto &i : conf.getPorts()) {
      auto name = i.getName();
      auto m = getPorts(name);
      m->update(i);
    }
  }

  if (conf.backendIsSet()) {
    auto m = getBackend();
    m->update(conf.getBackend());
  }
}

LbdsrJsonObject Lbdsr::toJsonObject() {
  LbdsrJsonObject conf;
  conf.setBase(Cube::to_json());

  conf.setFrontend(getFrontend()->toJsonObject());

  conf.setAlgorithm(getAlgorithm());

  for (auto &i : getPortsList()) {
    conf.addPorts(i->toJsonObject());
  }

  conf.setBackend(getBackend()->toJsonObject());

  return conf;
}

void Lbdsr::packet_in(Ports &port, polycube::service::PacketInMetadata &md,
                      const std::vector<uint8_t> &packet) {
  logger()->info("Packet received from port {0}", port.name());
}

std::string Lbdsr::getAlgorithm() {
  // This method retrieves the algorithm value.
  return this->algorithm_;
}

void Lbdsr::setAlgorithm(const std::string &value) {
  // This method set the algorithm value.
  this->algorithm_ = value;
}

void Lbdsr::replaceAll(std::string &str, const std::string &from,
                       const std::string &to) {
  if (from.empty())
    return;
  size_t start_pos = 0;
  while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length();  // In case 'to' contains 'from', like replacing
                               // 'x' with 'yx'
  }
}

std::string Lbdsr::getCode(bool first = false) {
  std::string code = lbdsr_code;

  std::string vip_hexbe_str_default_ = "0x6400000a";
  std::string mac_hexbe_str_default_ = "0xccbbaa010101";

  if (!first) {
    replaceAll(code, "_FRONTEND_IP", vip_hexbe_str_);
  } else {
    replaceAll(code, "_FRONTEND_IP", vip_hexbe_str_default_);
  }

  if (!first) {
    replaceAll(code, "_FRONTEND_MAC", mac_hexbe_str_);
  } else {
    replaceAll(code, "_FRONTEND_MAC", mac_hexbe_str_default_);
  }

  replaceAll(code, "_FRONTEND_PORT", std::to_string((uint16_t)frontend_port_));
  replaceAll(code, "_BACKEND_PORT", std::to_string((uint16_t)backend_port_));

  return code;
}

bool Lbdsr::reloadCode() {
  logger()->debug("reloadCode {0} ", is_code_changed_);
  if (is_code_changed_) {
    logger()->info("reloading code ...");
    reload(getCode());
    is_code_changed_ = false;
  }
  return true;
}

void Lbdsr::setFrontendPort(std::string portName, int index) {
  if (frontend_port_ != -1) {
    throw std::runtime_error("Frontend port already set");
  }

  frontend_port_ = index;
  frontend_port_str_ = portName;

  is_code_changed_ = true;
  reloadCode();
}

void Lbdsr::setBackendPort(std::string portName, int index) {
  if (backend_port_ != -1) {
    throw std::runtime_error("Backend port already set");
  }

  backend_port_ = index;
  backend_port_str_ = portName;

  is_code_changed_ = true;
  reloadCode();
}

void Lbdsr::rmPort(std::string portName) {
  if (portName == frontend_port_str_)
    rmFrontendPort(portName);
  else if (portName == backend_port_str_)
    rmBackendPort(portName);
  else
    throw std::runtime_error("port is not backend not frontend");
}

void Lbdsr::rmFrontendPort(std::string portName) {
  auto p = get_port(portName);

  int index = p->index();

  if (frontend_port_ == -1) {
    throw std::runtime_error("Frontend port not set");
  }

  frontend_port_ = -1;
  frontend_port_str_ = "";

  is_code_changed_ = true;
  reloadCode();
}

void Lbdsr::rmBackendPort(std::string portName) {
  auto p = get_port(portName);

  int index = p->index();

  if (backend_port_ == -1) {
    throw std::runtime_error("Backend port not set");
  }

  backend_port_ = -1;
  backend_port_str_ = "";

  is_code_changed_ = true;
  reloadCode();
}

void Lbdsr::setVipHexbe(std::string &value) {
  if (vip_hexbe_str_ != value) {
    vip_hexbe_str_ = value;
    is_code_changed_ = true;
    reloadCode();
  }
}

void Lbdsr::setMacHexbe(std::string &value) {
  if (mac_hexbe_str_ != value) {
    mac_hexbe_str_ = value;
    is_code_changed_ = true;
    reloadCode();
  }
}

std::shared_ptr<Ports> Lbdsr::getPorts(const std::string &name) {
  return get_port(name);
}

std::vector<std::shared_ptr<Ports>> Lbdsr::getPortsList() {
  return get_ports();
}

void Lbdsr::addPorts(const std::string &name, const PortsJsonObject &conf) {
  add_port<PortsJsonObject>(name, conf);
}

void Lbdsr::addPortsList(const std::vector<PortsJsonObject> &conf) {
  for (auto &i : conf) {
    std::string name_ = i.getName();
    addPorts(name_, i);
  }
}

void Lbdsr::replacePorts(const std::string &name, const PortsJsonObject &conf) {
  delPorts(name);
  std::string name_ = conf.getName();
  addPorts(name_, conf);
}

void Lbdsr::delPorts(const std::string &name) {
  // TODO: merge together those two functions?
  rmPort(name);
  remove_port(name);
}

void Lbdsr::delPortsList() {
  auto ports = get_ports();
  for (auto it : ports) {
    delPorts(it->name());
  }
}

std::shared_ptr<Frontend> Lbdsr::getFrontend() {
  if (frontend_ != nullptr) {
    return frontend_;
  } else {
    return std::make_shared<Frontend>(*this);
  }
}

void Lbdsr::addFrontend(const FrontendJsonObject &value) {
  logger()->debug("[Frontend] Received request to create new Frontend");
  frontend_ = std::make_shared<Frontend>(*this, value);
}

void Lbdsr::replaceFrontend(const FrontendJsonObject &conf) {
  delFrontend();
  addFrontend(conf);
}

void Lbdsr::delFrontend() {
  if (frontend_ != nullptr) {
    frontend_ = nullptr;
  } else {
    throw std::runtime_error("There is no frontend in this LBDSR");
  }
}

std::shared_ptr<Backend> Lbdsr::getBackend() {
  // This method retrieves the pointer to Backend object specified by its keys.
  if (backend_ != nullptr) {
    return backend_;
  } else {
    return std::make_shared<Backend>(*this);
  }
}

void Lbdsr::addBackend(const BackendJsonObject &value) {
  logger()->debug("[Backend] Received request to create new Backend");
  backend_ = std::make_shared<Backend>(*this, value);
}

void Lbdsr::replaceBackend(const BackendJsonObject &conf) {
  delBackend();
  addBackend(conf);
}

void Lbdsr::delBackend() {
  if (backend_ != nullptr) {
    backend_->delPoolList();
    backend_ = nullptr;
  } else {
    throw std::runtime_error("There is no backend in this LBDSR");
  }
}