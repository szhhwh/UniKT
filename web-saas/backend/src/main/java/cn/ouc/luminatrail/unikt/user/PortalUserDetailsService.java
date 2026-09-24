package cn.ouc.luminatrail.unikt.user;

import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

/** 按用户名加载门户用户为认证主体。 */
@Service
public class PortalUserDetailsService implements UserDetailsService {

    private final UserRepository users;

    public PortalUserDetailsService(UserRepository users) {
        this.users = users;
    }

    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        User user = users.findByUsername(username)
                .orElseThrow(() -> new UsernameNotFoundException("用户不存在"));
        return new PortalPrincipal(user.getId(), user.getUsername(), user.getPasswordHash(),
                user.getRole(), user.isActive());
    }
}
