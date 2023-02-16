package net.floodlightcontroller.linkdiscovery.web;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import net.floodlightcontroller.linkdiscovery.ILinkDiscovery.LinkDirection;
import net.floodlightcontroller.linkdiscovery.ILinkDiscovery.LinkType;
import net.floodlightcontroller.linkdiscovery.internal.LinkInfo;
import net.floodlightcontroller.linkdiscovery.ILinkDiscoveryService;
import net.floodlightcontroller.routing.Link;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.U64;
import org.restlet.resource.Get;
import org.restlet.resource.ServerResource;
public class LinksResource extends ServerResource {
    @Get("json")
    public Set<LinkWithType> retrieve() {
        ILinkDiscoveryService ld = (ILinkDiscoveryService)getContext().getAttributes().
                get(ILinkDiscoveryService.class.getCanonicalName());
        Map<Link, LinkInfo> links = new HashMap<Link, LinkInfo>();
        Set<LinkWithType> returnLinkSet = new HashSet<LinkWithType>();
        if (ld != null) {
            links.putAll(ld.getLinks());
            for (Link link: links.keySet()) {
                LinkInfo info = links.get(link);
                LinkType type = ld.getLinkType(link, info);
                if (type == LinkType.DIRECT_LINK || type == LinkType.TUNNEL) {
                    LinkWithType lwt;
                    DatapathId src = link.getSrc();
                    DatapathId dst = link.getDst();
                    OFPort srcPort = link.getSrcPort();
                    OFPort dstPort = link.getDstPort();
                    LinkInfo otherInfo = links.get(otherLink);
                    LinkType otherType = null;
                    if (otherInfo != null)
                        otherType = ld.getLinkType(otherLink, otherInfo);
                    if (otherType == LinkType.DIRECT_LINK ||
                            otherType == LinkType.TUNNEL) {
                        if ((src.getLong() < dst.getLong()) || (src.getLong() == dst.getLong()
                        		&& srcPort.getPortNumber() < dstPort.getPortNumber())) {
                            lwt = new LinkWithType(link,
                                    type,
                                    LinkDirection.BIDIRECTIONAL);
                            returnLinkSet.add(lwt);
                        }
                    } else {
                        lwt = new LinkWithType(link,
                                type,
                                LinkDirection.UNIDIRECTIONAL);
                        returnLinkSet.add(lwt);
                    }
                }
            }
        }
        return returnLinkSet;
    }
}
