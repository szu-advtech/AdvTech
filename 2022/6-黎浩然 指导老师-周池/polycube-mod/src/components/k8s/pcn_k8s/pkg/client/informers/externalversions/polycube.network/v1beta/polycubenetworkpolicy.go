/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Code generated by informer-gen. DO NOT EDIT.

package v1beta

import (
	time "time"

	polycubenetworkv1beta "github.com/polycube-network/polycube/src/components/k8s/pcn_k8s/pkg/apis/polycube.network/v1beta"
	versioned "github.com/polycube-network/polycube/src/components/k8s/pcn_k8s/pkg/client/clientset/versioned"
	internalinterfaces "github.com/polycube-network/polycube/src/components/k8s/pcn_k8s/pkg/client/informers/externalversions/internalinterfaces"
	v1beta "github.com/polycube-network/polycube/src/components/k8s/pcn_k8s/pkg/client/listers/polycube.network/v1beta"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtime "k8s.io/apimachinery/pkg/runtime"
	watch "k8s.io/apimachinery/pkg/watch"
	cache "k8s.io/client-go/tools/cache"
)

// PolycubeNetworkPolicyInformer provides access to a shared informer and lister for
// PolycubeNetworkPolicies.
type PolycubeNetworkPolicyInformer interface {
	Informer() cache.SharedIndexInformer
	Lister() v1beta.PolycubeNetworkPolicyLister
}

type polycubeNetworkPolicyInformer struct {
	factory          internalinterfaces.SharedInformerFactory
	tweakListOptions internalinterfaces.TweakListOptionsFunc
	namespace        string
}

// NewPolycubeNetworkPolicyInformer constructs a new informer for PolycubeNetworkPolicy type.
// Always prefer using an informer factory to get a shared informer instead of getting an independent
// one. This reduces memory footprint and number of connections to the server.
func NewPolycubeNetworkPolicyInformer(client versioned.Interface, namespace string, resyncPeriod time.Duration, indexers cache.Indexers) cache.SharedIndexInformer {
	return NewFilteredPolycubeNetworkPolicyInformer(client, namespace, resyncPeriod, indexers, nil)
}

// NewFilteredPolycubeNetworkPolicyInformer constructs a new informer for PolycubeNetworkPolicy type.
// Always prefer using an informer factory to get a shared informer instead of getting an independent
// one. This reduces memory footprint and number of connections to the server.
func NewFilteredPolycubeNetworkPolicyInformer(client versioned.Interface, namespace string, resyncPeriod time.Duration, indexers cache.Indexers, tweakListOptions internalinterfaces.TweakListOptionsFunc) cache.SharedIndexInformer {
	return cache.NewSharedIndexInformer(
		&cache.ListWatch{
			ListFunc: func(options v1.ListOptions) (runtime.Object, error) {
				if tweakListOptions != nil {
					tweakListOptions(&options)
				}
				return client.PolycubeV1beta().PolycubeNetworkPolicies(namespace).List(options)
			},
			WatchFunc: func(options v1.ListOptions) (watch.Interface, error) {
				if tweakListOptions != nil {
					tweakListOptions(&options)
				}
				return client.PolycubeV1beta().PolycubeNetworkPolicies(namespace).Watch(options)
			},
		},
		&polycubenetworkv1beta.PolycubeNetworkPolicy{},
		resyncPeriod,
		indexers,
	)
}

func (f *polycubeNetworkPolicyInformer) defaultInformer(client versioned.Interface, resyncPeriod time.Duration) cache.SharedIndexInformer {
	return NewFilteredPolycubeNetworkPolicyInformer(client, f.namespace, resyncPeriod, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc}, f.tweakListOptions)
}

func (f *polycubeNetworkPolicyInformer) Informer() cache.SharedIndexInformer {
	return f.factory.InformerFor(&polycubenetworkv1beta.PolycubeNetworkPolicy{}, f.defaultInformer)
}

func (f *polycubeNetworkPolicyInformer) Lister() v1beta.PolycubeNetworkPolicyLister {
	return v1beta.NewPolycubeNetworkPolicyLister(f.Informer().GetIndexer())
}
