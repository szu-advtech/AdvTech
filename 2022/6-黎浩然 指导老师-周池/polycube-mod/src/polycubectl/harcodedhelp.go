/*
 * Copyright 2017 The Polycube Authors
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

package main

import (
	"fmt"

	"github.com/polycube-network/polycube/src/polycubectl/cliargs"

	"github.com/ryanuber/columnize"
)

func printConnectHelp(args []string) {
	var output = [][]string{}

	output = append(output, []string{"keyword", "type", "description"})
	output = append(output, []string{"<from>", "cube:port", "Endpoint of the connection (E.g. br1:port1)"})
	output = append(output, []string{"<to>", "cube:port", "Endpoint of the connection (E.g. rt2:port2)"})

	config := columnize.DefaultConfig()
	config.NoTrim = true

	result := columnize.Format(output, config)
	Buffer += fmt.Sprintf("\n")
	Buffer += fmt.Sprintln(result)
	Buffer += fmt.Sprintf("\n")
	Buffer += fmt.Sprintf("Example:\n")
	Buffer += fmt.Sprintf(" %s br1:port1 rt2:port2\n\n", args[0])
}

func printAttachHelp() {
	var output = [][]string{}

	output = append(output, []string{"keyword", "type", "description"})
	output = append(output, []string{"<cube>", "cube", "Transparent cube to be attached (E.g. nat1)"})
	output = append(output, []string{"<port>", "cube:port or netdev", "Port of a cube or netdev to attach the transparent cube (E.g. rt2:port2 or eth0)"})
	output = append(output, []string{"position=value", "", "Position to place the cube. auto, first, last"})
	output = append(output, []string{"before=value", "", "Place the cube before another one"})
	output = append(output, []string{"after=value", "", "Place the cube after another one"})

	config := columnize.DefaultConfig()
	config.NoTrim = true

	result := columnize.Format(output, config)
	Buffer += fmt.Sprintf("\n")
	Buffer += fmt.Sprintln(result)
	Buffer += fmt.Sprintf("\n")
	Buffer += fmt.Sprintf("Example:\n")
	Buffer += fmt.Sprintf(" attach nat1 rt:port1\n\n")
}

func printDetachHelp() {
	var output = [][]string{}

	output = append(output, []string{"keyword", "type", "description"})
	output = append(output, []string{"<cube>", "cube", "Transparent cube to be detached (E.g. nat1)"})
	output = append(output, []string{"<port>", "cube:port or netdev", "Port of a cube or netdev to detach the transparent cube (E.g. rt2:port2 or eth0)"})

	config := columnize.DefaultConfig()
	config.NoTrim = true

	result := columnize.Format(output, config)
	Buffer += fmt.Sprintf("\n")
	Buffer += fmt.Sprintln(result)
	Buffer += fmt.Sprintf("\n")
	Buffer += fmt.Sprintf("Example:\n")
	Buffer += fmt.Sprintf(" detach nat1 rt:port1\n\n")
}

func printServicesHelp() {
	var output = [][]string{}
	output = append(output, []string{"keyword | type | description"})
	output = append(output, []string{"add", "command", getCommandDesc("add")})
	output = append(output, []string{"del", "command", getCommandDesc("del")})
	output = append(output, []string{"show", "command", getCommandDesc("show")})

	config := columnize.DefaultConfig()
	config.NoTrim = true

	result := columnize.Format(output, config)
	Buffer += fmt.Sprintf("\n")
	Buffer += fmt.Sprintln(result)
	Buffer += fmt.Sprintf("\n")
}

func printServicesAddHelp() {
	var output = [][]string{}
	output = append(output, []string{"Params:", "", ""})
	output = append(output, []string{"name=value", "string", "Name of the service"})
	output = append(output, []string{"uri=value", "string", "Path of the service (e.g. libpcn-bridge.so)"})
	output = append(output, []string{"type=value", "string", "Service type. lib or grpc"})

	config := columnize.DefaultConfig()
	config.NoTrim = true

	result := columnize.Format(output, config)
	Buffer += fmt.Sprintf("\n")
	Buffer += fmt.Sprintln(result)
	Buffer += fmt.Sprintf("\n")
	Buffer += fmt.Sprintf("Example:\n")
	Buffer += fmt.Sprintf(" services add name=bridge uri=libpcn-bridge.so type=lib\n\n")
}

func printServicesDelHelp() {
	var output = [][]string{}
	output = append(output, []string{"keyword", "type", "description"})
	output = append(output, []string{"<name>", "string", "Name of the service"})

	config := columnize.DefaultConfig()
	config.NoTrim = true

	result := columnize.Format(output, config)
	Buffer += fmt.Sprintf("\n")
	Buffer += fmt.Sprintln(result)
	Buffer += fmt.Sprintf("\n")
	Buffer += fmt.Sprintf("Example:\n")
	Buffer += fmt.Sprintf(" services del bridge\n\n")
}

func printServicesShowHelp() {
	var output = [][]string{}
	output = append(output, []string{"keyword", "type", "description"})
	output = append(output, []string{"<name>", "string", "Name of the service"})

	config := columnize.DefaultConfig()
	config.NoTrim = true

	result := columnize.Format(output, config)
	Buffer += fmt.Sprintf("\n")
	Buffer += fmt.Sprintln(result)
	Buffer += fmt.Sprintf("\n")
	Buffer += fmt.Sprintf("Example:\n")
	Buffer += fmt.Sprintf(" services show bridge\n\n")
}

func ServicesHelp(command string, args []string) {
	if command == "" {
		printServicesHelp()
	} else if command == cliargs.AddCommand {
		printServicesAddHelp()
	} else if command == cliargs.DelCommand {
		printServicesDelHelp()
	} else if command == cliargs.ShowCommand {
		printServicesShowHelp()
	}
}
